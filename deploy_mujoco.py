#!/usr/bin/env python3
"""Deploy G1 standing ONNX policy with classic MuJoCo (CPU).

The training environment uses **position** actuators; ``scene_29dof.xml`` uses
**motor** actuators. This script matches training physics by:
  1. Overriding the deploy model's ``dof_damping`` to the training values.
  2. Applying ``ctrl = kp * (q_target - q)`` as torque (kd=0, since damping
     is already in the model physics exactly as in training).

Observation exactly matches ``g1_standing._get_obs`` ``state`` (83-D):
  gyro(3) | gravity_local(3) | zeros_cmd(3) | leg_residual(12) | upper_abs(17)
  | joint_vel(29) | last_act(12) | zeros_phase(4)

Example::

  cd ~/Projects/mujoco_playground_g1_standing
  python deploy_mujoco.py --viewer
  python deploy_mujoco.py --config configs/g1.yaml --viewer --duration 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent

# ── Training nominal targets (g1_standing.default_config) ─────────────────────
LEG_POSE = np.array(
    [-0.2, 0.0, 0.0, 0.59, -0.34, 0.0,   # left  leg
     -0.2, 0.0, 0.0, 0.59, -0.34, 0.0],  # right leg
    dtype=np.float64,
)
UPPER_BODY_TARGET = np.zeros(17, dtype=np.float64)
ACTION_SCALE = 0.35

# ── Exact values from the training XML (g1_mjx_feetonly.xml) ──────────────────
# actuator_gainprm[:,0] for all 29 joints (same order as qpos[7:])
TRAIN_KP = np.array([
    75.0, 75.0, 75.0, 75.0, 20.0,  2.0,   # L hip(p/r/y), knee, ankle(p/r)
    75.0, 75.0, 75.0, 75.0, 20.0,  2.0,   # R hip(p/r/y), knee, ankle(p/r)
    75.0, 75.0, 75.0,               # waist (yaw, roll, pitch)
    75.0, 75.0, 75.0, 75.0,  2.0,  2.0,  2.0,  # L shoulder(p/r/y), elbow, wrist(r/p/y)
    75.0, 75.0, 75.0, 75.0,  2.0,  2.0,  2.0,  # R shoulder(p/r/y), elbow, wrist(r/p/y)
], dtype=np.float64)

# dof_damping[6:35] from the training XML — used to override the deploy model
# so the physics exactly matches training, letting us use kd=0 in the PD law.
TRAIN_DOF_DAMPING = np.array([
    2.0, 2.0, 2.0, 2.0, 1.0, 0.2,   # L hip(p/r/y), knee, ankle(p/r)
    2.0, 2.0, 2.0, 2.0, 1.0, 0.2,   # R hip(p/r/y), knee, ankle(p/r)
    2.0, 2.0, 2.0,                   # waist
    2.0, 2.0, 2.0, 2.0, 0.2, 0.2, 0.2,  # L shoulder/elbow/wrists
    2.0, 2.0, 2.0, 2.0, 0.2, 0.2, 0.2,  # R shoulder/elbow/wrists
], dtype=np.float64)

NOISE_SCALES = {
    "gyro": 0.1,
    "gravity": 0.02,
    "joint_pos": 0.01,
    "joint_vel": 0.3,
}

# Same spirit as ``g1_standing._get_termination`` (torso up-axis + base height).
_TORSO_BODY_CANDIDATES = ("torso_link", "pelvis")


def detect_fall(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    height_threshold: float,
) -> tuple[bool, str]:
  """Return (fallen, reason). Reasons: nan, torso_up, low_height."""
  if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
    return True, "nan_or_nonfinite_state"

  torso_id = -1
  for name in _TORSO_BODY_CANDIDATES:
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if torso_id >= 0:
      break
  if torso_id < 0:
    return False, ""

  # Body z-axis in world frame (column 2 of 3x3 xmat); upright → positive Z component.
  z_world = np.array(data.xmat[torso_id], dtype=np.float64).reshape(3, 3)[:, 2]
  if float(z_world[2]) < 0.0:
    return True, "torso_z_world<=0"

  h = float(data.qpos[2])
  if h < height_threshold:
    return True, f"low_base_height (z={h:.3f} < {height_threshold})"

  return False, ""


def _load_yaml(path: Path) -> dict:
  try:
    import yaml  # type: ignore
  except ImportError as e:
    raise RuntimeError("Install PyYAML for --config (pip install pyyaml)") from e
  with open(path, encoding="utf-8") as f:
    return yaml.safe_load(f) or {}


def _sensor_slice(model: mujoco.MjModel, name: str) -> tuple[int, int]:
  sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
  if sid < 0:
    raise KeyError(f"Sensor not found: {name!r}")
  return int(model.sensor_adr[sid]), int(model.sensor_dim[sid])


def build_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pose: np.ndarray,
    last_act: np.ndarray,
    noise_level: float,
    rng: np.random.Generator,
) -> np.ndarray:

  adr, dim = _sensor_slice(model, "imu-pelvis-angular-velocity")
  gyro = np.array(data.sensordata[adr : adr + dim], dtype=np.float64)
  sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu_in_pelvis")
  R = data.site_xmat[sid].reshape(3, 3)
  gravity = R.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
  qj = np.array(data.qpos[7:36], dtype=np.float64)
  qd = np.array(data.qvel[6:35], dtype=np.float64)
  leg_residual = qj[:12] - target_pose[:12]
  upper_abs = qj[12:]

  obs = np.concatenate([
      gyro,                         # 3 (raw)
      gravity,                      # 3 (raw)
      np.zeros(3, dtype=np.float64),  # cmd (velocity commands = 0 at deploy)
      leg_residual,                 # 12
      upper_abs,                    # 17
      qd,                           # 29 (raw)
      last_act.astype(np.float64),  # 12
      np.zeros(4, dtype=np.float64),  # gait phase = 0 (not used for standing)
  ])
  assert obs.shape == (83,), f"obs shape {obs.shape} != (83,)"
  return obs.astype(np.float32)


def reset_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pose: np.ndarray,
    base_height: float,
) -> None:
  data.qpos[:] = 0.0
  data.qvel[:] = 0.0
  data.ctrl[:] = 0.0
  data.qpos[2] = base_height
  data.qpos[3] = 1.0  # quaternion w=1 → identity rotation (upright)
  data.qpos[7:36] = target_pose
  mujoco.mj_forward(model, data)


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--onnx", type=Path, default=REPO_ROOT / "model" / "g1_standing_policy.onnx")
  p.add_argument("--onnx_meta", type=Path, default=None)
  p.add_argument("--scene", type=Path, default=REPO_ROOT / "resources" / "g1" / "scene_29dof.xml")
  p.add_argument("--duration",    type=float, default=60.0)
  p.add_argument("--sim_dt",      type=float, default=None)
  p.add_argument("--decimation",  type=int,   default=10)
  p.add_argument("--action_scale",type=float, default=ACTION_SCALE)
  p.add_argument("--obs_noise",   type=float, default=0.0)
  p.add_argument("--viewer",      action="store_true")
  p.add_argument("--config",      type=Path,  default=None)
  p.add_argument(
      "--fall_height",
      type=float,
      default=None,
      help="Base height (qpos[2]) below this counts as fallen; default 0.45 (training env).",
  )
  p.add_argument(
      "--continue_after_fall",
      action="store_true",
      help="Keep simulating after a fall (default: stop and exit after reporting).",
  )
  args = p.parse_args()

  cfg: dict = {}
  if args.config is not None:
    cfg = _load_yaml(args.config.expanduser().resolve())

  onnx_path  = Path(cfg.get("onnx_policy_path", args.onnx)).expanduser().resolve()
  scene_path = Path(cfg.get("scene_xml_path",   args.scene)).expanduser().resolve()

  if not onnx_path.is_file():
    print(f"ONNX not found: {onnx_path}", file=sys.stderr); sys.exit(1)
  if not scene_path.is_file():
    print(f"Scene XML not found: {scene_path}", file=sys.stderr); sys.exit(1)

  meta_path = args.onnx_meta or onnx_path.with_suffix(".onnx.json")
  if meta_path.is_file():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(meta.get("obs_dim", 83)) != 83:
      print(f"Warning: meta obs_dim={meta.get('obs_dim')} expected 83", file=sys.stderr)

  try:
    import onnxruntime as ort
  except ImportError as e:
    print("pip install onnxruntime", file=sys.stderr)
    raise SystemExit(1) from e

  session  = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
  in_name  = session.get_inputs()[0].name
  out_name = session.get_outputs()[0].name

  model = mujoco.MjModel.from_xml_path(str(scene_path))
  if args.sim_dt is not None:
    model.opt.timestep = float(args.sim_dt)

  if model.nu != 29 or model.nq != 36:
    print(f"Expected nu=29 nq=36; got nu={model.nu} nq={model.nq}", file=sys.stderr)
    sys.exit(1)

  # Match training physics: per-joint damping in model + Kp position servo in control.
  model.dof_damping[6:35] = TRAIN_DOF_DAMPING
  data = mujoco.MjData(model)

  # KP: full 29-joint vector (from training XML actuator_gainprm).
  # Config can override only the 12 leg kps via 'kps' key.
  kp = TRAIN_KP.copy()
  if "kps" in cfg:
    kp_leg = np.array(cfg["kps"], dtype=np.float64)
    if kp_leg.size != 12:
      print("config 'kps' must have 12 entries", file=sys.stderr); sys.exit(1)
    kp[:12] = kp_leg

  leg_pose = np.array(cfg.get("leg_pose", LEG_POSE.tolist()), dtype=np.float64)
  upper    = np.array(cfg.get("upper_body_target", UPPER_BODY_TARGET.tolist()), dtype=np.float64)
  if leg_pose.size != 12 or upper.size != 17:
    print("leg_pose must be length 12, upper_body_target length 17", file=sys.stderr)
    sys.exit(1)
  target_pose  = np.concatenate([leg_pose, upper])
  base_height  = float(cfg.get("base_height", 0.79))
  action_scale = float(cfg.get("action_scale", args.action_scale))

  reset_state(model, data, target_pose, base_height)
  rng = np.random.default_rng(int(cfg.get("seed", 0)))

  dt     = model.opt.timestep
  n_steps = int(args.duration / dt)
  dec    = max(1, int(args.decimation))
  fall_h = float(cfg.get("fall_height", 0.45))
  if args.fall_height is not None:
    fall_h = float(args.fall_height)
  # Ignore spurious triggers during the first control transients.
  fall_check_start = max(1, int(round(0.05 / dt)))

  # Actuator i directly drives joint i+1 (same order as qpos[7:]).
  # Verified: actuator[0] → left_hip_pitch_joint → slot 0, etc.
  train_qpos_range = np.array([
      [-2.5307, 2.8798], [-0.5236, 2.9671], [-2.7576, 2.7576],   # L hip p/r/y
      [-0.0873, 2.8798], [-0.8727, 0.5236], [-0.2618, 0.2618],   # L knee, ankle p/r
      [-2.5307, 2.8798], [-0.5236, 2.9671], [-2.7576, 2.7576],   # R hip p/r/y
      [-0.0873, 2.8798], [-0.8727, 0.5236], [-0.2618, 0.2618],   # R knee, ankle p/r
      [-2.6180, 2.6180], [-0.5200, 0.5200], [-0.5200, 0.5200],   # waist y/r/p
      [-3.0892, 2.6704], [-1.5882, 2.2515], [-2.6180, 2.6180], [-1.0472, 2.0944],  # L arm
      [-1.9722, 1.9722], [-1.6144, 1.6144], [-1.6144, 1.6144],   # L wrist
      [-3.0892, 2.6704], [-2.2515, 1.5882], [-2.6180, 2.6180], [-1.0472, 2.0944],  # R arm
      [-1.9722, 1.9722], [-1.6144, 1.6144], [-1.6144, 1.6144],   # R wrist
  ], dtype=np.float64)

  fallen = False
  fall_reason = ""
  fall_step: int | None = None  # first fall; sim time then = (fall_step + 1) * dt
  last_step = -1

  def step_body(step_i: int) -> bool:
    """Return True to stop the simulation loop."""
    nonlocal fallen, fall_reason, fall_step
    if step_i < fall_check_start:
      return False
    if fallen and args.continue_after_fall:
      return False
    if fallen:
      return True
    f, why = detect_fall(model, data, height_threshold=fall_h)
    if f:
      fallen = True
      fall_reason = why
      fall_step = step_i
      return not args.continue_after_fall
    return False
  # Desired qpos command updated at policy rate (decimated control ticks).
  q_des = target_pose.copy()
  last_act = np.zeros(12, dtype=np.float32)
  if args.viewer:
    step_i = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
      while viewer.is_running() and step_i < n_steps:
        if step_i % dec == 0:
          obs = build_obs(model, data, target_pose, last_act, args.obs_noise, rng)
          act = session.run([out_name], {in_name: obs.reshape(1, -1)})[0].reshape(-1)
          act = np.clip(act.astype(np.float32), -1.0, 1.0)
          last_act = act

          # Target joint positions: legs = nominal + policy residual, upper = nominal.
          q_des = target_pose.copy()
          q_des[:12] += action_scale * act.astype(np.float64)

          # Clip to training joint range limits.
          q_des = np.clip(q_des, train_qpos_range[:, 0], train_qpos_range[:, 1])

        for i in range(model.nu):
          jid  = int(model.actuator_trnid[i, 0])
          qadr = int(model.jnt_qposadr[jid])
          q    = float(data.qpos[qadr])
          data.ctrl[i] = kp[i] * (q_des[i] - q)

        mujoco.mj_step(model, data)
        last_step = step_i
        if step_body(step_i):
          break
        step_i += 1
        viewer.sync()
  else:
    for step_i in range(n_steps):
      if step_i % dec == 0:
        obs = build_obs(model, data, target_pose, last_act, args.obs_noise, rng)
        act = session.run([out_name], {in_name: obs.reshape(1, -1)})[0].reshape(-1)
        act = np.clip(act.astype(np.float32), -1.0, 1.0)
        last_act = act

        # Target joint positions: legs = nominal + policy residual, upper = nominal.
        q_des = target_pose.copy()
        q_des[:12] += action_scale * act.astype(np.float64)

        # Clip to training joint range limits.
        q_des = np.clip(q_des, train_qpos_range[:, 0], train_qpos_range[:, 1])

      for i in range(model.nu):
        jid  = int(model.actuator_trnid[i, 0])
        qadr = int(model.jnt_qposadr[jid])
        q    = float(data.qpos[qadr])
        data.ctrl[i] = kp[i] * (q_des[i] - q)
      mujoco.mj_step(model, data)
      last_step = step_i
      if step_body(step_i):
        break

  if fall_step is not None:
    t_stand_until = (fall_step + 1) * dt
  else:
    t_stand_until = (last_step + 1) * dt
  tail = f"fall: {fall_reason}" if fallen else "no fall"
  if args.continue_after_fall and fallen:
    tail += "; continued after fall"
  print(f"Standing until t = {t_stand_until:.4f} s (sim); {tail}.")


if __name__ == "__main__":
  main()
