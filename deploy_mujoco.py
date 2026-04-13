#!/usr/bin/env python3
"""Deploy G1 standing ONNX policy with classic MuJoCo (CPU).

Observation matches ``g1_standing._get_obs`` ``state`` (83-D):
  gyro(3) | gravity(3) | cmd(3) | leg_residual(12) | upper_abs(17)
  | joint_vel(29) | last_act(12) | phase(4)

Example::

  python deploy_mujoco.py configs/g1.yaml --viewer
  python deploy_mujoco.py configs/g1.yaml --viewer --duration 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent

# GLFW key codes for special keys (letters use their ASCII uppercase value).
_KEY_MAP: dict[str, int] = {
    "LEFT": 263, "RIGHT": 262, "UP": 265, "DOWN": 264,
    "SPACE": 32,
    "ENTER": 257, "TAB": 258,
}


def parse_key(key_str: str) -> int:
    """Return the GLFW keycode for a key name (e.g. 'LEFT', 'A', 'SPACE')."""
    s = key_str.strip().upper()
    if s in _KEY_MAP:
        return _KEY_MAP[s]
    if len(s) == 1:
        return ord(s)
    raise ValueError(f"Unknown key: {key_str!r}. Use a letter, or one of {list(_KEY_MAP)}")


# Joint position limits from the training XML (29 joints, [lo, hi]).
QPOS_RANGE = np.array([
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

_TORSO_BODY_CANDIDATES = ("torso_link", "pelvis")


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion
    return np.array([
        2.0 * (-qz * qx + qw * qy),
        -2.0 * (qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz),
    ])


def pd_control(target_q: np.ndarray, q: np.ndarray, kp: np.ndarray) -> np.ndarray:
    """Position servo with damping baked into the model (kd = 0)."""
    return (target_q - q) * kp


def build_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_pose: np.ndarray,
    last_act: np.ndarray,
) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu-pelvis-angular-velocity")
    adr, dim = int(model.sensor_adr[sid]), int(model.sensor_dim[sid])
    gyro = np.array(data.sensordata[adr : adr + dim], dtype=np.float64)

    quat = np.array(data.qpos[3:7], dtype=np.float64)
    quat /= np.linalg.norm(quat) + 1e-8
    gravity = get_gravity_orientation(quat)

    qj = np.array(data.qpos[7:36], dtype=np.float64)
    qd = np.array(data.qvel[6:35], dtype=np.float64)
    leg_residual = qj[:12] - target_pose[:12]
    upper_abs    = qj[12:]

    obs = np.concatenate([
        gyro,                          # 3
        gravity,                       # 3
        np.zeros(3),                   # cmd = 0 (standing)
        leg_residual,                  # 12
        upper_abs,                     # 17
        qd,                            # 29
        last_act.astype(np.float64),   # 12
        np.zeros(4),                   # gait phase = 0 (not used)
    ])
    assert obs.shape == (83,), f"obs shape {obs.shape} != (83,)"
    return obs.astype(np.float32)


def detect_fall(
    model: mujoco.MjModel,
    data: mujoco.MjData,
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

    z_world = np.array(data.xmat[torso_id]).reshape(3, 3)[:, 2]
    if float(z_world[2]) < 0.0:
        return True, "torso_z_world<=0"

    h = float(data.qpos[2])
    if h < height_threshold:
        return True, f"low_base_height (z={h:.3f} < {height_threshold})"

    return False, ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", type=str, help="path to YAML config (e.g. configs/g1.yaml)")
    parser.add_argument("--viewer",              action="store_true")
    parser.add_argument("--continue_after_fall", action="store_true")
    parser.add_argument("--duration",            type=float, default=None,
                        help="override simulation_duration from config")
    args = parser.parse_args()

    with open(args.config_file, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    onnx_path  = (REPO_ROOT / cfg["onnx_policy_path"]).resolve()
    scene_path = (REPO_ROOT / cfg["scene_xml_path"]).resolve()

    if not onnx_path.is_file():
        print(f"ONNX not found: {onnx_path}", file=sys.stderr); sys.exit(1)
    if not scene_path.is_file():
        print(f"Scene XML not found: {scene_path}", file=sys.stderr); sys.exit(1)

    simulation_duration = args.duration or float(cfg["simulation_duration"])
    simulation_dt       = float(cfg["simulation_dt"])
    control_decimation  = int(cfg["control_decimation"])

    kps         = np.array(cfg["kps"],         dtype=np.float64)
    dof_damping = np.array(cfg["dof_damping"], dtype=np.float64)

    leg_pose    = np.array(cfg["leg_pose"],          dtype=np.float64)
    upper_body  = np.array(cfg["upper_body_target"], dtype=np.float64)
    target_pose = np.concatenate([leg_pose, upper_body])  # 29-D

    action_scale = float(cfg["action_scale"])
    base_height  = float(cfg["base_height"])
    fall_height  = float(cfg.get("fall_height", 0.45))

    try:
        import onnxruntime as ort
    except ImportError:
        print("pip install onnxruntime", file=sys.stderr); sys.exit(1)

    session  = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name  = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.timestep   = simulation_dt
    model.dof_damping[6:35] = dof_damping
    data = mujoco.MjData(model)

    # Reset to nominal standing pose.
    data.qpos[:] = 0.0
    data.qpos[2] = base_height
    data.qpos[3] = 1.0        # unit quaternion → upright
    data.qpos[7:36] = target_pose
    mujoco.mj_forward(model, data)

    n_steps          = int(simulation_duration / simulation_dt)
    fall_check_start = max(1, int(round(0.05 / simulation_dt)))

    # ── Disturbances ──────────────────────────────────────────────────────────
    # Each entry: {keycode, body_id, wrench (6-D force+torque, world frame), duration}
    disturbance_bindings: list[dict] = []
    for d in cfg.get("disturbances", []):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, d["body"])
        if body_id < 0:
            print(f"Warning: disturbance body {d['body']!r} not found — skipping", file=sys.stderr)
            continue
        wrench = np.zeros(6, dtype=np.float64)
        wrench[:3] = d.get("force",  [0.0, 0.0, 0.0])
        wrench[3:] = d.get("torque", [0.0, 0.0, 0.0])
        disturbance_bindings.append({
            "keycode":  parse_key(d["key"]),
            "body_id":  body_id,
            "body":     d["body"],
            "wrench":   wrench,
            "duration": float(d["duration"]),
            "label":    d.get("label", d["key"]),
        })

    if disturbance_bindings and args.viewer:
        print("\nDisturbance keys:")
        for b in disturbance_bindings:
            f, t = b["wrench"][:3], b["wrench"][3:]
            print(f"  [{b['label']:8s}]  body={b['body']}  "
                  f"F={f.tolist()}  T={t.tolist()}  dur={b['duration']}s")
        print()

    # Maps body_id → (wrench, end_sim_time) for currently active disturbances.
    active_disturbances: dict[int, tuple[np.ndarray, float]] = {}

    def key_callback(keycode: int) -> None:
        for b in disturbance_bindings:
            if keycode == b["keycode"]:
                end_time = data.time + b["duration"]
                active_disturbances[b["body_id"]] = (b["wrench"], end_time)
                print(f"[t={data.time:.2f}s] Push '{b['label']}' on {b['body']}  "
                      f"F={b['wrench'][:3].tolist()} N  for {b['duration']}s")

    # ── Simulation state ───────────────────────────────────────────────────────
    action    = np.zeros(12, dtype=np.float32)
    q_des     = target_pose.copy()
    fallen    = False
    fall_reason = ""
    fall_step   = None
    last_step   = -1

    def step(step_i: int) -> bool:
        """Advance simulation by one step. Returns True to stop the loop."""
        nonlocal action, q_des, fallen, fall_reason, fall_step, last_step

        if step_i % control_decimation == 0:
            obs    = build_obs(model, data, target_pose, action)
            action = np.clip(
                session.run([out_name], {in_name: obs.reshape(1, -1)})[0].reshape(-1),
                -1.0, 1.0,
            ).astype(np.float32)
            q_des = target_pose.copy()
            q_des[:12] += action_scale * action.astype(np.float64)
            q_des = np.clip(q_des, QPOS_RANGE[:, 0], QPOS_RANGE[:, 1])

        data.ctrl[:] = pd_control(q_des, np.array([
            data.qpos[int(model.jnt_qposadr[int(model.actuator_trnid[i, 0])])]
            for i in range(model.nu)
        ]), kps)

        # Apply disturbance forces (cleared each step so they don't accumulate).
        data.xfrc_applied[:] = 0.0
        expired = [bid for bid, (_, end_t) in active_disturbances.items()
                   if data.time > end_t]
        for bid in expired:
            del active_disturbances[bid]
        for bid, (wrench, _) in active_disturbances.items():
            data.xfrc_applied[bid] = wrench

        mujoco.mj_step(model, data)
        last_step = step_i

        if step_i >= fall_check_start:
            f, why = detect_fall(model, data, fall_height)
            if f and not fallen:
                fallen, fall_reason, fall_step = True, why, step_i
            if fallen and not args.continue_after_fall:
                return True

        return False

    if args.viewer:
        step_i = 0
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            while viewer.is_running() and step_i < n_steps:
                if step(step_i):
                    break
                step_i += 1
                viewer.sync()
    else:
        for step_i in range(n_steps):
            if step(step_i):
                break

    t      = ((fall_step if fall_step is not None else last_step) + 1) * simulation_dt
    status = f"fall: {fall_reason}" if fallen else "no fall"
    print(f"Standing until t = {t:.4f} s; {status}.")


if __name__ == "__main__":
    main()
