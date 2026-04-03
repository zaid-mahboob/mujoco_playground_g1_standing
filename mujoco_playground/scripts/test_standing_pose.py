"""
Standalone pose feasibility tester for G1 standing.

Loads the G1 model with plain mujoco (no JAX/MJX), precomputes QP-based
gravity compensation at startup (same as g1_standing.py), then runs a
live PD controller to hold the chosen standing pose.

Usage (conda unitree-rl env):
    python scripts/test_standing_pose.py
    python scripts/test_standing_pose.py --no_viewer --duration 10

Interpretation:
  - Stable hold → pose is feasible with these gains.
  - Drift / fall → pose or PD gains need adjustment.
"""

import argparse
import pathlib
import time

import mujoco
import mujoco.viewer
import numpy as np

# ---------------------------------------------------------------------------
# Paths (all relative to this script's location)
# ---------------------------------------------------------------------------
_REPO    = pathlib.Path(__file__).resolve().parents[1]
_XML_DIR = _REPO / "mujoco_playground/_src/locomotion/g1/xmls"
_ASSETS_DIR = _XML_DIR / "assets"
_MENAGERIE  = _REPO / "mujoco_playground/external_deps/mujoco_menagerie/unitree_g1"

_SCENE_XML  = _XML_DIR / "scene_mjx_feetonly_flat_terrain.xml"


def _load_assets():
    """Read all XML + mesh files from the two asset directories into a bytes dict.

    mujoco.MjModel.from_xml_string resolves <mesh file="..."> names by looking
    up the basename in this dict, bypassing relative-path issues.
    """
    assets = {}
    for directory in [_XML_DIR, _ASSETS_DIR, _MENAGERIE, _MENAGERIE / "assets"]:
        if not directory.exists():
            continue
        for f in directory.iterdir():
            if f.is_file():
                assets[f.name] = f.read_bytes()
    return assets


# ---------------------------------------------------------------------------
# Standing pose — must match g1_standing.py default_config()
# ---------------------------------------------------------------------------
LEG_POSE = np.array([
    -0.2, 0.0, 0.0, 0.59, -0.34, 0.0,   # left leg
    -0.2, 0.0, 0.0, 0.59, -0.34, 0.0,   # right leg
], dtype=np.float64)

QP_GAMMA = 0.5  # symmetric stance; must match g1_standing qp_gamma
BASE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # identity — upright

HEIGHT_TARGET = 0.715  # target pelvis height (m) — from RL reward config

# PD gains — must match actuator_gainprm / dof_damping in g1_mjx_feetonly.xml
# Order: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll (×2 legs)
LEG_KP = np.array([50, 50, 20, 50, 5, 5,
                   50, 50, 20, 50, 5, 5], dtype=np.float64)
LEG_KD = np.array([15, 15, 10, 15, 5, 5,
                   15, 15, 10, 15, 5, 5], dtype=np.float64)

# Order: waist(3), left_arm(7), right_arm(7)
UPPER_KP = np.array([200, 245, 210,
                     200, 200, 100, 160, 80, 80, 80,
                     200, 200, 100, 160, 80, 80, 80], dtype=np.float64)
UPPER_KD = np.array([12.0, 18.6, 17.3,
                     16.0, 16.0,  9.0, 14.0, 6.0, 6.0, 6.0,
                     16.0, 16.0,  9.0, 14.0, 6.0, 6.0, 6.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# QP-based compensation (identical to g1_standing.py)
# ---------------------------------------------------------------------------
def compute_qp_compensation(model: mujoco.MjModel,
                             leg_pose: np.ndarray,
                             height: float,
                             gamma: float) -> np.ndarray:
    """Precompute feedforward torques: tau = h_gravity - J_contact.T @ w_qp.

    Solved once at startup (CPU, plain mujoco).
    Returns (12,) torques for leg joints.
    """
    try:
        import cvxpy as cp
        _has_cp = True
    except ImportError:
        _has_cp = False

    data = mujoco.MjData(model)
    nv   = model.nv

    qpos = np.zeros(model.nq)
    qpos[2]   = height
    qpos[3]   = 1.0          # quaternion w — upright
    qpos[7:19] = leg_pose
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    data.qacc[:] = 0.0
    h = np.empty(nv)
    mujoco.mj_rne(model, data, 0, h)
    h_base   = h[:6]
    h_joints = h[6:]

    if not _has_cp:
        print("[Warning] cvxpy not found — using pure gravity compensation.")
        return h_joints[:12].copy()

    left_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    JpL, JrL = np.zeros((3, nv)), np.zeros((3, nv))
    JpR, JrR = np.zeros((3, nv)), np.zeros((3, nv))
    mujoco.mj_jacBody(model, data, JpL, JrL, left_id)
    mujoco.mj_jacBody(model, data, JpR, JrR, right_id)

    H_b = np.zeros((6, 12))
    H_b[:, 0:3]  = JpL[:, :6].T;  H_b[:, 3:6]  = JrL[:, :6].T
    H_b[:, 6:9]  = JpR[:, :6].T;  H_b[:, 9:12] = JrR[:, :6].T

    Fz_total = float(np.sum(model.body_mass)) * float(np.abs(model.opt.gravity[2]))

    Rbw_L = data.xmat[left_id].reshape(3, 3).T
    Rbw_R = data.xmat[right_id].reshape(3, 3).T
    T_L, T_R = np.zeros((6, 12)), np.zeros((6, 12))
    T_L[:3, :3]   = Rbw_L;  T_L[3:6, 3:6]   = Rbw_L
    T_R[:3, 6:9]  = Rbw_R;  T_R[3:6, 9:12]  = Rbw_R

    w = cp.Variable(12)
    wL_b, wR_b = T_L @ w, T_R @ w
    mu, cop_x, cop_y = 0.6, 0.06, 0.03

    objective = cp.Minimize(
        1e6 * cp.sum_squares(H_b @ w - h_base)
        + 1e6 * cp.sum_squares(w[2] + w[8] - Fz_total)
        + 1e4 * cp.sum_squares((1 - gamma) * w[2] - gamma * w[8])
        + 5e2 * cp.sum_squares(cp.hstack([w[0], w[1], w[6], w[7]]))
        + 1e2 * cp.sum_squares(cp.hstack([w[5], w[11]]))
        + 1e-6 * cp.sum_squares(w)
    )
    constraints = [
        wL_b[2] >= 0,   wR_b[2] >= 0,
        wL_b[0] <=  mu * wL_b[2],  -wL_b[0] <= mu * wL_b[2],
        wL_b[1] <=  mu * wL_b[2],  -wL_b[1] <= mu * wL_b[2],
        wL_b[4] <= cop_x * wL_b[2], -wL_b[4] <= cop_x * wL_b[2],
        wL_b[3] <= cop_y * wL_b[2], -wL_b[3] <= cop_y * wL_b[2],
        wR_b[0] <=  mu * wR_b[2],  -wR_b[0] <= mu * wR_b[2],
        wR_b[1] <=  mu * wR_b[2],  -wR_b[1] <= mu * wR_b[2],
        wR_b[4] <= cop_x * wR_b[2], -wR_b[4] <= cop_x * wR_b[2],
        wR_b[3] <= cop_y * wR_b[2], -wR_b[3] <= cop_y * wR_b[2],
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8,
               max_iter=100_000, verbose=False)

    if w.value is None:
        print("[Warning] QP failed — using pure gravity compensation.")
        return h_joints[:12].copy()

    wv  = w.value
    fL, mL = wv[:3], wv[3:6]
    fR, mR = wv[6:9], wv[9:12]
    tau = (h_joints
           - JpL[:, 6:].T @ fL - JrL[:, 6:].T @ mL
           - JpR[:, 6:].T @ fR - JrR[:, 6:].T @ mR)
    return tau[:12].copy()


def compute_gravity_torques(model: mujoco.MjModel,
                             leg_pose: np.ndarray,
                             height: float) -> np.ndarray:
    """Gravity-only (mj_rne, qvel=0) at the nominal pose — for live correction."""
    data = mujoco.MjData(model)
    qpos = np.zeros(model.nq)
    qpos[2]   = height
    qpos[3]   = 1.0
    qpos[7:19] = leg_pose
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    data.qacc[:] = 0.0
    h = np.empty(model.nv)
    mujoco.mj_rne(model, data, 0, h)
    return h[6:18].copy()


# ---------------------------------------------------------------------------
# Online controller (called every sim step)
# ---------------------------------------------------------------------------
def step_controller(model: mujoco.MjModel,
                    data: mujoco.MjData,
                    target_qpos: np.ndarray,
                    comp_tau: np.ndarray,
                    G_nominal: np.ndarray) -> None:
    """
    Compute and apply ctrl each step.

    Feedforward:
        tau_ff = comp_tau + (G_current - G_nominal)
               = (G_nominal - J.T @ w) + (G_current - G_nominal)
               = G_current - J.T @ w

    G_current uses mj_rne with qvel=0 at current qpos (live gravity correction).
    The result is converted to a position offset and added to the target position
    before being sent to the position actuators.
    """
    q    = data.qpos[7:]    # (29,)
    qdot = data.qvel[6:]    # (29,)

    # Live gravity correction — zero qvel in a temporary copy (no side-effects).
    tmp = mujoco.MjData(model)
    tmp.qpos[:] = data.qpos
    tmp.qvel[:] = 0.0
    mujoco.mj_forward(model, tmp)
    tmp.qacc[:] = 0.0
    h = np.empty(model.nv)
    mujoco.mj_rne(model, tmp, 0, h)
    G_current = h[6:18]

    # Feedforward torque → position offset (same conversion as RL env)
    tau_ff     = comp_tau + G_current - G_nominal
    comp_off   = np.clip((tau_ff + LEG_KD * qdot[:12]) / LEG_KP, -0.5, 0.5)

    # Desired ctrl (position targets) for each actuator
    leg_ctrl   = target_qpos[:12] + comp_off
    upper_ctrl = target_qpos[12:]   # zeros — hold at 0

    data.ctrl[:29] = np.concatenate([leg_ctrl, upper_ctrl])


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def print_diagnostics(model: mujoco.MjModel,
                       data: mujoco.MjData,
                       target_qpos: np.ndarray,
                       last_print: list) -> None:
    t = data.time
    if t - last_print[0] < 1.0:
        return
    last_print[0] = t

    pelvis_h  = data.qpos[2]
    leg_err   = np.abs(data.qpos[7:19] - target_qpos[:12])
    torso_id  = model.body("torso_link").id
    # Column 2 of xmat is the body z-axis in world frame
    torso_z_world = data.xmat[torso_id].reshape(3, 3)[:, 2]
    tilt_deg  = np.degrees(np.arccos(np.clip(torso_z_world[2], -1.0, 1.0)))

    print(f"  t={t:5.1f}s | pelvis_h={pelvis_h:.3f}m | "
          f"leg_err_max={leg_err.max():.3f}rad | tilt={tilt_deg:.1f}°")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_viewer", action="store_true",
                        help="Run headless (print diagnostics only).")
    parser.add_argument("--duration",  type=float, default=30.0,
                        help="Simulation duration in seconds (headless only).")
    args = parser.parse_args()

    # --- Load model ---
    if not _SCENE_XML.exists():
        raise FileNotFoundError(f"Scene XML not found: {_SCENE_XML}")
    if not _MENAGERIE.exists():
        raise FileNotFoundError(
            f"MuJoCo Menagerie not found at {_MENAGERIE}.\n"
            "Run the training script once (it auto-downloads) or clone it manually."
        )

    print(f"Loading model from {_SCENE_XML} …")
    assets = _load_assets()
    model  = mujoco.MjModel.from_xml_string(_SCENE_XML.read_text(), assets=assets)
    data   = mujoco.MjData(model)
    print(f"  nq={model.nq}  nv={model.nv}  nu={model.nu}")

    leg_pose  = LEG_POSE.copy()
    gamma     = QP_GAMMA
    base_quat = BASE_QUAT.copy()

    # Full 29-DOF target: legs at pose, upper body at zero
    target_qpos          = np.zeros(29)
    target_qpos[:12]     = leg_pose

    # --- Precompute (once at startup) ---
    print(f"\nPrecomputing QP compensation (gamma={gamma}) …")
    comp_tau  = compute_qp_compensation(model, leg_pose, HEIGHT_TARGET, gamma)
    G_nominal = compute_gravity_torques(model,  leg_pose, HEIGHT_TARGET)
    print(f"  comp_tau  = {np.round(comp_tau, 2)}")
    print(f"  G_nominal = {np.round(G_nominal, 2)}")

    # --- Initial state ---
    mujoco.mj_resetData(model, data)
    data.qpos[2]   = HEIGHT_TARGET
    data.qpos[3:7] = base_quat
    data.qpos[7:]  = target_qpos
    data.ctrl[:29] = target_qpos          # warm-start actuators
    mujoco.mj_forward(model, data)

    last_print = [0.0]

    # --- Run ---
    if args.no_viewer:
        print(f"\nRunning headless for {args.duration}s …")
        while data.time < args.duration:
            step_controller(model, data, target_qpos, comp_tau, G_nominal)
            mujoco.mj_step(model, data)
            print_diagnostics(model, data, target_qpos, last_print)
        print("Done.")
    else:
        print("\nOpening MuJoCo viewer — close window to exit.\n")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0.0, 0.0, 0.75]
            viewer.cam.distance  = 2.5
            viewer.cam.elevation = -15
            while viewer.is_running():
                step_start = time.perf_counter()
                step_controller(model, data, target_qpos, comp_tau, G_nominal)
                mujoco.mj_step(model, data)
                print_diagnostics(model, data, target_qpos, last_print)
                viewer.sync()
                # Real-time pacing
                dt_wall = time.perf_counter() - step_start
                if dt_wall < model.opt.timestep:
                    time.sleep(model.opt.timestep - dt_wall)


if __name__ == "__main__":
    main()
