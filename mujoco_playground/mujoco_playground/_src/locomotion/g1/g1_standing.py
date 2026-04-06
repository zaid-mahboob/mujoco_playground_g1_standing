# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Standing task for Unitree G1."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import base as g1_base
from mujoco_playground._src.locomotion.g1 import g1_constants as consts


def _com_world_from_fk_np(
    com_subtree: np.ndarray,
    foot_xpos: np.ndarray,
    foot_world: np.ndarray,
) -> np.ndarray:
  """World CoM: foot_world + (com_FK - foot_FK)."""
  return foot_world + (com_subtree - foot_xpos)


def _com_world_from_fk_jp(
    com_subtree: jax.Array,
    foot_xpos: jax.Array,
    foot_world: jax.Array,
) -> jax.Array:
  return foot_world + (com_subtree - foot_xpos)


def _get_com_position_np(
    mj_model,
    joint_angles: np.ndarray,
    stance_foot: str,
    pelvis_id: int,
    left_foot_body_id: int,
    right_foot_body_id: int,
    foot_world_left: np.ndarray,
    foot_world_right: np.ndarray,
    base_quat: Optional[np.ndarray] = None,
    fk_base_pos: Optional[np.ndarray] = None,
) -> np.ndarray:
  """World CoM from FK with fixed base xyz (not measured on robot), quat, encoders."""
  import mujoco as _mj

  if fk_base_pos is None:
    fk_base_pos = np.array([0.0, 0.0, 2.0], dtype=np.float64)
  d = _mj.MjData(mj_model)
  d.qpos[:3] = fk_base_pos
  if base_quat is not None:
    d.qpos[3:7] = base_quat
  else:
    d.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
  n_j = int(joint_angles.shape[0])
  d.qpos[7 : 7 + n_j] = joint_angles
  _mj.mj_forward(mj_model, d)
  com_pos = d.subtree_com[pelvis_id].copy()
  if stance_foot == "left":
    foot_id = left_foot_body_id
    foot_world = foot_world_left
  else:
    foot_id = right_foot_body_id
    foot_world = foot_world_right
  return _com_world_from_fk_np(com_pos, d.xpos[foot_id].copy(), foot_world)


def _compute_gravity_torques(
    mj_model,
    leg_pose: np.ndarray,
    height: float,
) -> np.ndarray:
  """Gravity-only joint torques (mj_rne, qvel=0) for the 12 leg DOFs."""
  import mujoco as _mj

  data = _mj.MjData(mj_model)
  qpos = np.zeros(mj_model.nq)
  qpos[2] = height
  qpos[3] = 1.0
  qpos[7:19] = leg_pose
  data.qpos[:] = qpos
  data.qvel[:] = 0.0
  _mj.mj_forward(mj_model, data)
  data.qacc[:] = 0.0
  h = np.empty(mj_model.nv)
  _mj.mj_rne(mj_model, data, 0, h)
  return h[6:18].astype(np.float32)


def _compute_qp_compensation_torques(
    mj_model,
    leg_pose: np.ndarray,
    height: float,
    gamma: float = 0.5,
) -> np.ndarray:
  """QP contact wrenches at nominal stand; tau = h - J'w. Fallback: mj_rne legs."""
  import mujoco as _mj

  try:
    import cvxpy as cp
    _has_cp = True
  except ImportError:
    _has_cp = False

  data = _mj.MjData(mj_model)
  nv = mj_model.nv
  qpos = np.zeros(mj_model.nq)
  qpos[2] = height
  qpos[3] = 1.0
  qpos[7:19] = leg_pose
  data.qpos[:] = qpos
  data.qvel[:] = 0.0
  _mj.mj_forward(mj_model, data)
  data.qacc[:] = 0.0
  h = np.empty(nv)
  _mj.mj_rne(mj_model, data, 0, h)
  h_base, h_joints = h[:6], h[6:]

  if not _has_cp:
    return h_joints[:12].astype(np.float32)

  left_id = _mj.mj_name2id(
      mj_model, _mj.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"
  )
  right_id = _mj.mj_name2id(
      mj_model, _mj.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"
  )
  JpL, JrL = np.zeros((3, nv)), np.zeros((3, nv))
  JpR, JrR = np.zeros((3, nv)), np.zeros((3, nv))
  _mj.mj_jacBody(mj_model, data, JpL, JrL, left_id)
  _mj.mj_jacBody(mj_model, data, JpR, JrR, right_id)

  JpL_b, JrL_b = JpL[:, :6], JrL[:, :6]
  JpR_b, JrR_b = JpR[:, :6], JrR[:, :6]
  JpL_j, JrL_j = JpL[:, 6:], JrL[:, 6:]
  JpR_j, JrR_j = JpR[:, 6:], JrR[:, 6:]

  H_b = np.zeros((6, 12))
  H_b[:, 0:3] = JpL_b.T
  H_b[:, 3:6] = JrL_b.T
  H_b[:, 6:9] = JpR_b.T
  H_b[:, 9:12] = JrR_b.T

  Fz_total = float(np.sum(mj_model.body_mass)) * float(
      np.abs(mj_model.opt.gravity[2])
  )

  Rbw_L = data.xmat[left_id].reshape(3, 3).T
  Rbw_R = data.xmat[right_id].reshape(3, 3).T
  T_L, T_R = np.zeros((6, 12)), np.zeros((6, 12))
  T_L[:3, :3] = Rbw_L
  T_L[3:6, 3:6] = Rbw_L
  T_R[:3, 6:9] = Rbw_R
  T_R[3:6, 9:12] = Rbw_R

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
      wL_b[2] >= 0,
      wR_b[2] >= 0,
      wL_b[0] <= mu * wL_b[2],
      -wL_b[0] <= mu * wL_b[2],
      wL_b[1] <= mu * wL_b[2],
      -wL_b[1] <= mu * wL_b[2],
      wL_b[4] <= cop_x * wL_b[2],
      -wL_b[4] <= cop_x * wL_b[2],
      wL_b[3] <= cop_y * wL_b[2],
      -wL_b[3] <= cop_y * wL_b[2],
      wR_b[0] <= mu * wR_b[2],
      -wR_b[0] <= mu * wR_b[2],
      wR_b[1] <= mu * wR_b[2],
      -wR_b[1] <= mu * wR_b[2],
      wR_b[4] <= cop_x * wR_b[2],
      -wR_b[4] <= cop_x * wR_b[2],
      wR_b[3] <= cop_y * wR_b[2],
      -wR_b[3] <= cop_y * wR_b[2],
  ]
  prob = cp.Problem(objective, constraints)
  prob.solve(
      solver=cp.OSQP,
      eps_abs=1e-8,
      eps_rel=1e-8,
      max_iter=100000,
      verbose=False,
  )

  if w.value is None:
    return h_joints[:12].astype(np.float32)

  wv = w.value
  fL, mL = wv[:3], wv[3:6]
  fR, mR = wv[6:9], wv[9:12]
  tau_joints = (
      h_joints
      - JpL_j.T @ fL
      - JrL_j.T @ mL
      - JpR_j.T @ fR
      - JrR_j.T @ mR
  )
  return tau_joints[:12].astype(np.float32)


LEG_POSE_LIBRARY = np.array(
    [
        [-0.20, 0.00, 0.00, 0.59, -0.34, 0.00, -0.20, 0.00, 0.00, 0.59, -0.34, 0.00],
        [-0.20, 0.10, 0.08, 0.59, -0.34, 0.05, -0.20, -0.10, -0.08, 0.59, -0.34, -0.05],
        [-0.22, 0.20, 0.10, 0.65, -0.38, 0.08, -0.22, -0.20, -0.10, 0.65, -0.38, -0.08],
        [-0.25, 0.32, 0.14, 0.80, -0.48, 0.13, -0.25, -0.32, -0.14, 0.80, -0.48, -0.13],
        [-0.28, 0.42, 0.18, 0.95, -0.57, 0.17, -0.28, -0.42, -0.18, 0.95, -0.57, -0.17],
        [-0.10, 0.00, 0.00, 0.25, -0.14, 0.00, -0.10, 0.00, 0.00, 0.25, -0.14, 0.00],
        [-0.10, 0.12, 0.06, 0.25, -0.14, 0.05, -0.10, -0.12, -0.06, 0.25, -0.14, -0.05],
        [-0.20, 0.00, 0.00, 1.00, -0.60, 0.00, -0.20, 0.00, 0.00, 1.00, -0.60, 0.00],
        [-0.25, 0.30, 0.12, 1.00, -0.60, 0.12, -0.25, -0.30, -0.12, 1.00, -0.60, -0.12],
        [-0.20, 0.15, 0.20, 0.59, -0.34, 0.06, -0.20, -0.15, -0.20, 0.59, -0.34, -0.06],
        [-0.22, 0.25, 0.10, 0.78, -0.46, 0.10, -0.18, -0.08, -0.04, 0.45, -0.26, -0.03],
        [-0.25, 0.08, 0.06, 0.85, -0.50, 0.03, -0.12, -0.15, -0.08, 0.30, -0.17, -0.06],
        [-0.12, 0.35, 0.10, 0.30, -0.17, 0.14, -0.12, -0.35, -0.10, 0.30, -0.17, -0.14],
        [-0.18, 0.12, 0.10, 0.55, -0.32, 0.05, -0.18, -0.12, -0.10, 0.55, -0.32, -0.05],
        [-0.22, 0.28, 0.12, 0.72, -0.43, 0.11, -0.22, -0.28, -0.12, 0.72, -0.43, -0.11],
    ],
    dtype=np.float64,
)


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.002,
      episode_length=3000,
      action_repeat=1,
      action_scale=0.35,
      restricted_joint_range=False,
      soft_joint_pos_limit_factor=0.95,
      # left_leg(6) + right_leg(6) — [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
      leg_pose=[
          -0.2, 0.0, 0.0, 0.59, -0.34, 0.0,   # left leg
          -0.2, 0.0, 0.0, 0.59, -0.34, 0.0,   # right leg
      ],
      # Pelvis orientation [w, x, y, z] — identity = upright (0° roll/pitch/yaw).
      base_quat=[1.0, 0.0, 0.0, 0.0],
      qp_gamma=0.5,
      upper_body_target=[0.0] * 17,  # waist(3) + arms(14)
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.01,
              joint_vel=0.3,
              gravity=0.02,
              linvel=0.05,
              gyro=0.1,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              orientation=-2.0,
              base_height=-1.0,
              lin_vel_z=-0.3,
              ang_vel_xy=-0.5,
              base_linvel_xy=-0.8,
              base_angvel_yaw=-0.3,
              com_stability=-0.8,
              pose=-0.0,
              joint_vel=-0.01,
              dof_pos_limits=-1.0,
              action_rate=-0.03,
              torques=0.0,
              energy=0.0,
              dof_acc=0.0,
              collision=-0.0,
              contact_force=-0.05,
              foot_contact_symmetry=-0.2,
              foot_flatness=-0.2,
              foot_slip=-0.05,
              alive=1.0,
              still_bonus=0.5,
              termination=-100.0,
          ),
          base_height_target=0.763,
          max_contact_force=500.0,
      ),
      push_config=config_dict.create(
          enable=True,
          interval_range=[3.0, 7.0],
          magnitude_range=[0.05, 1.0],
      ),
      joint_disturbance=config_dict.create(
          enable=True,
          interval_range=[1.5, 4.0],
          duration_range=[2, 8],
          magnitude_range=[0.02, 0.12],
      ),
      upper_body_pose_disturbance=config_dict.create(
          enable=True,
          # Uniform magnitude then ± sign; result is clamped per joint so
          # upper_body_target[j] + offset stays within mj jnt_range (hard limits).
          # Example (nominal 0, g1_mjx_feetonly.xml): waist_yaw ∈ [-2.618, 2.618] rad
          # ⇒ offset ∈ [-2.618, 2.618]; waist_roll / waist_pitch ∈ [-0.52, 0.52] rad
          # ⇒ offset ∈ [-0.52, 0.52] each (tighter than magnitude_range caps those joints).
          magnitude_range=[0.03, 0.35],
      ),
      curriculum_config=config_dict.create(
          enable=True,
          # Curriculum progress reaches 1.0 by this many env steps.
          ramp_steps=10_000_000,
          # Keep all disturbances off before this global step.
          disturbance_warmup_steps=1_000_000,
          use_performance_gate=True,
          ema_alpha=0.02,
          target_episode_fraction=0.8,
          target_fall_rate=0.2,
          # Push curriculum.
          push_mag_scale_start=0.25,
          push_mag_scale_end=1.0,
          push_interval_scale_start=2.0,  # larger => less frequent early
          push_interval_scale_end=1.0,
          # Leg-joint disturbance curriculum.
          joint_mag_scale_start=0.25,
          joint_mag_scale_end=1.0,
          joint_interval_scale_start=2.0,
          joint_interval_scale_end=1.0,
          # Upper-body pose disturbance curriculum (per-episode offset magnitude).
          # Multiplies upper_body_pose_disturbance.magnitude_range when sampling each episode.
          upper_mag_scale_start=0.20,
          upper_mag_scale_end=1.0,
      ),
      com_pid_config=config_dict.create(
          enable=True,
          kp=[1000.0, 550.0, 0.0],
          ki=[50.0, 50.0, 0.0],
          kd=[100.0, 250.0, 0.0],
          ankle_pitch_scale=1.0,
          ankle_roll_scale=1.0,
          hip_pitch_scale=0.0,
          hip_roll_scale=0.0,
          max_force=500.0,
          max_integral=0.1,
      ),
      support_height_perturb_range=[-0.01, 0.02],
      action_latency_config=config_dict.create(
          enable=True,
          min_steps=1,
          max_steps=3,
      ),
      standing_pose_randomization=config_dict.create(
          enable=True,
      ),
      # CoM FK only uses IMU quat + joints; base translation is arbitrary (deploy: [0,0,2]).
      com_fk_base_pos=[0.0, 0.0, 2.0],
      impl="warp",
      naconmax=8 * 8192,
      njmax=29 * 2 + 8 * 4,
  )


class Standing(g1_base.G1Env):
  """Keep G1 standing robustly under disturbances."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("knees_bent").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("knees_bent").qpos[7:])

    self._leg_pose = jp.array(self._config.leg_pose)
    self._base_quat = jp.array(self._config.base_quat)
    self._upper_target = jp.array(self._config.upper_body_target)

    # Leg kp for torque → position offset (position actuators, DOFs 6–17).
    self._leg_kp = jp.array(self._mj_model.actuator_gainprm[:12, 0])
    self._leg_kd = jp.array(self._mj_model.dof_damping[6:18])

    height = float(self._config.reward_config.base_height_target)
    default_leg_pose_np = np.array(self._config.leg_pose, dtype=np.float64)
    gamma = float(self._config.qp_gamma)

    if self._config.standing_pose_randomization.enable:
      self._leg_pose_library = jp.array(LEG_POSE_LIBRARY)
    else:
      self._leg_pose_library = jp.array(default_leg_pose_np[jp.newaxis, :])

    import mujoco as _mj

    pelvis_id = self._mj_model.body("pelvis").id
    lf_id = self._mj_model.body("left_ankle_roll_link").id
    rf_id = self._mj_model.body("right_ankle_roll_link").id
    upper_np = np.array(self._config.upper_body_target, dtype=np.float64)
    self._comp_tau_library = []
    self._G_standing_library = []
    self._target_com_left_library = []
    self._target_com_right_library = []

    for pose_np in np.array(self._leg_pose_library, dtype=np.float64):
      self._comp_tau_library.append(
          _compute_qp_compensation_torques(self._mj_model, pose_np, height, gamma)
      )
      self._G_standing_library.append(
          _compute_gravity_torques(self._mj_model, pose_np, height)
      )

      joint_nom = np.concatenate([pose_np, upper_np])
      tmp = _mj.MjData(self._mj_model)
      qpos_nom = np.zeros(self._mj_model.nq)
      qpos_nom[2] = height
      qpos_nom[3] = 1.0
      qpos_nom[7 : 7 + joint_nom.shape[0]] = joint_nom
      tmp.qpos[:] = qpos_nom
      _mj.mj_forward(self._mj_model, tmp)
      fw_l = tmp.xpos[lf_id].copy()
      fw_r = tmp.xpos[rf_id].copy()

      _bq = np.array([1.0, 0.0, 0.0, 0.0])
      self._target_com_left_library.append(
          _get_com_position_np(
              self._mj_model,
              joint_nom,
              "left",
              pelvis_id,
              lf_id,
              rf_id,
              fw_l,
              fw_r,
              base_quat=_bq,
              fk_base_pos=np.array(self._config.com_fk_base_pos, dtype=np.float64),
          )
      )
      self._target_com_right_library.append(
          _get_com_position_np(
              self._mj_model,
              joint_nom,
              "right",
              pelvis_id,
              lf_id,
              rf_id,
              fw_l,
              fw_r,
              base_quat=_bq,
              fk_base_pos=np.array(self._config.com_fk_base_pos, dtype=np.float64),
          )
      )

    self._comp_tau_library = jp.array(np.array(self._comp_tau_library))
    self._G_standing_library = jp.array(np.array(self._G_standing_library))
    self._target_com_left_library = jp.array(np.array(self._target_com_left_library))
    self._target_com_right_library = jp.array(np.array(self._target_com_right_library))
    self._comp_tau = self._comp_tau_library[0]
    self._G_standing = self._G_standing_library[0]
    joint_nom = np.concatenate([default_leg_pose_np, upper_np])
    tmp = _mj.MjData(self._mj_model)
    qpos_nom = np.zeros(self._mj_model.nq)
    qpos_nom[2] = height
    qpos_nom[3] = 1.0
    qpos_nom[7 : 7 + joint_nom.shape[0]] = joint_nom
    tmp.qpos[:] = qpos_nom
    _mj.mj_forward(self._mj_model, tmp)
    fw_l = tmp.xpos[lf_id].copy()
    fw_r = tmp.xpos[rf_id].copy()
    self._pelvis_body_id = pelvis_id
    self._left_foot_body_id = lf_id
    self._right_foot_body_id = rf_id
    self._foot_world_left = jp.array(fw_l)
    self._foot_world_right = jp.array(fw_r)
    self._foot_world_mid = (self._foot_world_left + self._foot_world_right) / 2.0
    fk_base_np = np.array(self._config.com_fk_base_pos, dtype=np.float64)
    # Nominal upright CoM anchors (target: identity quat; same stance foot as current at runtime).
    _bq = np.array([1.0, 0.0, 0.0, 0.0])
    self._target_com_left = jp.array(
        _get_com_position_np(
            self._mj_model,
            joint_nom,
            "left",
            pelvis_id,
            lf_id,
            rf_id,
            fw_l,
            fw_r,
            base_quat=_bq,
            fk_base_pos=fk_base_np,
        )
    )
    self._target_com_right = jp.array(
        _get_com_position_np(
            self._mj_model,
            joint_nom,
            "right",
            pelvis_id,
            lf_id,
            rf_id,
            fw_l,
            fw_r,
            base_quat=_bq,
            fk_base_pos=fk_base_np,
        )
    )

    cpid = self._config.com_pid_config
    self._com_kp = jp.array(cpid.kp, dtype=jp.float32)
    self._com_ki = jp.array(cpid.ki, dtype=jp.float32)
    self._com_kd = jp.array(cpid.kd, dtype=jp.float32)
    self._com_ankle_pitch_scale = float(cpid.ankle_pitch_scale)
    self._com_ankle_roll_scale = float(cpid.ankle_roll_scale)
    self._com_hip_pitch_scale = float(cpid.hip_pitch_scale)
    self._com_hip_roll_scale = float(cpid.hip_roll_scale)
    self._com_max_force = float(cpid.max_force)
    self._com_max_integral = float(cpid.max_integral)

    self._leg_indices = jp.arange(12)
    self._upper_indices = jp.arange(12, 29)
    # hip_pitch indices (0, 6) and hip_roll indices (1, 7) — used for reference
    self._hip_indices = jp.array([0, 1, 6, 7])
    self._knee_indices = jp.array([3, 9])

    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor
    # Feasible additive offset for upper-body disturbance: nominal + offset ∈ [lower, upper].
    self._upper_offset_min = self._lowers[self._upper_indices] - self._upper_target
    self._upper_offset_max = self._uppers[self._upper_indices] - self._upper_target

    self._pelvis_imu_site_id = self._mj_model.site("imu_in_pelvis").id
    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )

    self._feet_floor_found_sensor = [
        self._mj_model.sensor(foot_geom + "_floor_found").id
        for foot_geom in ["left_foot", "right_foot"]
    ]
    self._right_foot_left_foot_found_sensor = self._mj_model.sensor(
        "right_foot_left_foot_found"
    ).id
    self._left_foot_right_shin_found_sensor = self._mj_model.sensor(
        "left_foot_right_shin_found"
    ).id
    self._right_foot_left_shin_found_sensor = self._mj_model.sensor(
        "right_foot_left_shin_found"
    ).id
    self._left_hand_left_thigh_found_sensor = self._mj_model.sensor(
        "left_hand_left_thigh_found"
    ).id
    self._right_hand_right_thigh_found_sensor = self._mj_model.sensor(
        "right_hand_right_thigh_found"
    ).id
    self._left_foot_upvector_adr = self._mj_model.sensor_adr[
        self._mj_model.sensor("left_foot_upvector").id
    ]
    self._right_foot_upvector_adr = self._mj_model.sensor_adr[
        self._mj_model.sensor("right_foot_upvector").id
    ]

  @property
  def action_size(self) -> int:
    """Policy controls legs only (12 joints)."""
    return 12

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.1, maxval=0.1)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)

    # Upright pelvis, then apply random yaw on top.
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-0.3, maxval=0.3)
    yaw_quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    qpos = qpos.at[3:7].set(math.quat_mul(self._base_quat, yaw_quat))

    rng, key = jax.random.split(rng)
    height_perturb = jax.random.uniform(
        key,
        (1,),
        minval=self._config.support_height_perturb_range[0],
        maxval=self._config.support_height_perturb_range[1],
    )
    qpos = qpos.at[2].set(qpos[2] + height_perturb[0])

    rng, key = jax.random.split(rng)
    pose_idx = jax.random.randint(
        key, (), minval=0, maxval=self._leg_pose_library.shape[0]
    )
    sampled_leg_pose = self._leg_pose_library[pose_idx]
    sampled_comp_tau = self._comp_tau_library[pose_idx]
    sampled_g_standing = self._G_standing_library[pose_idx]
    sampled_target_com_left = self._target_com_left_library[pose_idx]
    sampled_target_com_right = self._target_com_right_library[pose_idx]

    target_pose = self._default_pose
    target_pose = target_pose.at[self._leg_indices].set(sampled_leg_pose)
    target_pose = target_pose.at[self._upper_indices].set(self._upper_target)

    rng, key = jax.random.split(rng)
    target_pose += jax.random.uniform(key, (29,), minval=-0.02, maxval=0.02)
    target_pose = jp.clip(target_pose, self._lowers, self._uppers)
    qpos = qpos.at[7:].set(target_pose)

    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.1, maxval=0.1)
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=target_pose,
        impl=self.mjx_model.impl.value,
        naconmax=self._config.naconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    rng, push_rng = jax.random.split(rng)
    push_interval_steps = self._sample_interval_steps(
        push_rng, self._config.push_config.interval_range
    )
    rng, jd_rng = jax.random.split(rng)
    disturb_interval_steps = self._sample_interval_steps(
        jd_rng, self._config.joint_disturbance.interval_range
    )
    rng, latency_rng = jax.random.split(rng)
    latency_steps = jp.where(
        self._config.action_latency_config.enable,
        jax.random.randint(
            latency_rng,
            (),
            self._config.action_latency_config.min_steps,
            self._config.action_latency_config.max_steps + 1,
        ),
        jp.array(0, dtype=jp.int32),
    )

    info = {
        "rng": rng,
        "step": jp.array(0, dtype=jp.int32),
        "global_step": jp.array(0, dtype=jp.int32),
        "episode_step": jp.array(0, dtype=jp.int32),
        "episode_len_ema": jp.array(0.0),
        "fall_rate_ema": jp.array(1.0),
        "pose_idx": pose_idx.astype(jp.int32),
        "target_pose": target_pose,
        "comp_tau": sampled_comp_tau,
        "G_standing": sampled_g_standing,
        "target_com_left": sampled_target_com_left,
        "target_com_right": sampled_target_com_right,
        "com_integral": jp.zeros(3),
        "prev_com": (sampled_target_com_left + sampled_target_com_right) / 2.0,
        "com_vel_filtered": jp.zeros(3),
        "last_act": jp.zeros(self.action_size),
        "last_last_act": jp.zeros(self.action_size),
        "action_history": jp.zeros((4, self.action_size)),
        "action_latency_steps": latency_steps.astype(jp.int32),
        "motor_targets": target_pose,
        "push_step": jp.array(0, dtype=jp.int32),
        "push_interval_steps": push_interval_steps,
        "disturb_step": jp.array(0, dtype=jp.int32),
        "disturb_interval_steps": disturb_interval_steps,
        "disturb_time_left": jp.array(0, dtype=jp.int32),
        "disturb_joint": jp.array(0, dtype=jp.int32),
        "disturb_value": jp.array(0.0),
        # One joint + offset for whole episode; resampled when done (see step).
        "upper_episode_joint": jp.array(0, dtype=jp.int32),
        "upper_episode_offset": jp.array(0.0),
        "push_profile_id": jp.array(0, dtype=jp.int32),
        "prev_foot_xy": data.site_xpos[self._feet_site_id, :2],
    }

    metrics = {
        f"reward/{k}": jp.zeros(())
        for k in self._config.reward_config.scales.keys()
    }
    metrics["disturbance_profile_id"] = jp.zeros(())

    contact = self._get_foot_contact(data)
    obs = self._get_obs(data, info, contact)
    return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

  def _qpos_com_fk(self, data: mjx.Data) -> jax.Array:
    """FK state as on hardware: fixed base xyz (not from SLAM), IMU quat, joint encoders."""
    base_xyz = jp.asarray(self._config.com_fk_base_pos, dtype=data.qpos.dtype)
    return (
        jp.zeros_like(data.qpos)
        .at[0:3].set(base_xyz)
        .at[3:7].set(data.qpos[3:7])
        .at[7:].set(data.qpos[7:])
    )

  def _com_stance_from_mjx_fwd(
      self, fwd: mjx.Data
  ) -> tuple[jax.Array, jax.Array]:
    """Lower foot by z on same FK snapshot; world CoM via foot anchor (no measured CoM)."""
    zl = fwd.xpos[self._left_foot_body_id, 2]
    zr = fwd.xpos[self._right_foot_body_id, 2]
    use_left = zl <= zr
    foot_x = jp.where(
        use_left,
        fwd.xpos[self._left_foot_body_id],
        fwd.xpos[self._right_foot_body_id],
    )
    foot_w = jp.where(
        use_left, self._foot_world_left, self._foot_world_right
    )
    com_w = _com_world_from_fk_jp(
        fwd.subtree_com[self._pelvis_body_id], foot_x, foot_w
    )
    return use_left, com_w

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    rng = state.info["rng"]
    rng, push_rng, disturb_rng = jax.random.split(rng, 3)

    data = state.data
    push = self._sample_push(push_rng, state.info)
    qvel = data.qvel.at[:2].set(data.qvel[:2] + push)
    data = data.replace(qvel=qvel)

    action_history = jp.concatenate(
        [action[jp.newaxis, :], state.info["action_history"][:-1]], axis=0
    )
    latency_steps = state.info["action_latency_steps"]
    delayed_action = action_history[latency_steps]

    leg_targets = (
        state.info["target_pose"][self._leg_indices]
        + delayed_action * self._config.action_scale
    )
    motor_targets = state.info["target_pose"].at[self._leg_indices].set(leg_targets)
    motor_targets = motor_targets.at[self._upper_indices].set(self._upper_target)

    # QP gravity feedforward (nominal) + online bias correction → Δq ≈ τ/kp.
    qpos_com_fk = self._qpos_com_fk(data)
    fwd_com_fk = mjx.forward(
        self.mjx_model,
        data.replace(qpos=qpos_com_fk, qvel=jp.zeros_like(data.qvel)),
    )
    g_current = fwd_com_fk.qfrc_bias[6:18]
    comp_tau = state.info["comp_tau"] + g_current - state.info["G_standing"]
    comp_offset = jp.clip(comp_tau / self._leg_kp, -0.5, 0.5)
    motor_targets = motor_targets.at[self._leg_indices].add(comp_offset)

    # CoM PID → ankle/hip torque hints → same Δq mapping.
    if self._config.com_pid_config.enable:
      use_left_stance, current_com = self._com_stance_from_mjx_fwd(fwd_com_fk)
      target_com = jp.where(
          use_left_stance, state.info["target_com_left"], state.info["target_com_right"]
      )
      com_vel_raw = (current_com - state.info["prev_com"]) / self.dt
      com_vel_filtered = (
          0.15 * com_vel_raw + 0.85 * state.info["com_vel_filtered"]
      )
      com_error = target_com - current_com
      new_com_integral = jp.clip(
          state.info["com_integral"] + com_error * self.dt,
          -self._com_max_integral,
          self._com_max_integral,
      )
      f_com = jp.clip(
          self._com_kp * com_error
          + self._com_ki * new_com_integral
          - self._com_kd * com_vel_filtered,
          -self._com_max_force,
          self._com_max_force,
      )
      r_com = current_com - self._foot_world_mid
      m_pitch = -r_com[2] * f_com[0]
      m_roll = r_com[2] * f_com[1]
      m_hip_roll = self._com_hip_roll_scale * r_com[2] * f_com[1]
      m_hip_pitch = -self._com_hip_pitch_scale * r_com[2] * f_com[0]
      tau_com = (
          jp.zeros(12)
          .at[4].add(self._com_ankle_pitch_scale * m_pitch / 2)   # left ankle_pitch
          .at[5].add(self._com_ankle_roll_scale * m_roll / 2)     # left ankle_roll
          .at[10].add(self._com_ankle_pitch_scale * m_pitch / 2)  # right ankle_pitch
          .at[11].add(self._com_ankle_roll_scale * m_roll / 2)    # right ankle_roll
          .at[1].add(m_hip_roll / 2)    # left hip_roll (index 1)
          .at[7].add(-m_hip_roll / 2)   # right hip_roll (index 7, opposite sign)
          .at[0].add(m_hip_pitch / 2)   # left hip_pitch (index 0)
          .at[6].add(m_hip_pitch / 2)   # right hip_pitch (index 6)
      )
      com_offset = jp.clip(tau_com / self._leg_kp, -0.3, 0.3)
      motor_targets = motor_targets.at[self._leg_indices].add(com_offset)
    else:
      new_com_integral = state.info["com_integral"]
      com_vel_filtered = state.info["com_vel_filtered"]
      current_com = state.info["prev_com"]

    motor_targets = self._apply_joint_disturbance(
        motor_targets, disturb_rng, state.info
    )
    motor_targets = self._apply_upper_body_episode_offset(motor_targets, state.info)
    motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)

    data = mjx_env.step(self.mjx_model, data, motor_targets, self.n_substeps)
    contact = self._get_foot_contact(data)
    done = self._get_termination(data)

    obs = self._get_obs(data, state.info, contact)
    rewards = self._get_reward(data, delayed_action, state.info, done, contact)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = sum(rewards.values()) * self.dt

    state.info["step"] += 1
    state.info["global_step"] += 1
    state.info["episode_step"] += 1
    state.info["push_step"] += 1
    state.info["disturb_step"] += 1
    rng, resample_rng = jax.random.split(rng)
    new_uj, new_uo = self._sample_upper_body_episode_offset(resample_rng, state.info)
    disturb_on = self._disturbances_enabled(state.info)
    state.info["upper_episode_joint"] = jp.where(
        done,
        jp.where(disturb_on, new_uj, jp.array(0, dtype=jp.int32)),
        state.info["upper_episode_joint"],
    )
    state.info["upper_episode_offset"] = jp.where(
        done,
        jp.where(disturb_on, new_uo, jp.array(0.0)),
        state.info["upper_episode_offset"],
    )
    state.info["rng"] = rng
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = delayed_action
    state.info["action_history"] = action_history
    state.info["motor_targets"] = motor_targets
    state.info["prev_foot_xy"] = data.site_xpos[self._feet_site_id, :2]

    if self._config.com_pid_config.enable:
      state.info["com_integral"] = new_com_integral
      state.info["prev_com"] = current_com
      state.info["com_vel_filtered"] = com_vel_filtered

    state.info["disturb_time_left"] = jp.maximum(
        state.info["disturb_time_left"] - 1, 0
    )
    alpha = self._config.curriculum_config.ema_alpha
    ep_len_frac = (
        state.info["episode_step"].astype(jp.float32) / self._config.episode_length
    )
    done_f = done.astype(jp.float32)
    # episode_len_ema: only update at episode end (fraction of max episode survived).
    state.info["episode_len_ema"] = jp.where(
        done_f > 0,
        (1.0 - alpha) * state.info["episode_len_ema"] + alpha * ep_len_frac,
        state.info["episode_len_ema"],
    )
    # fall_rate_ema: update every step toward done_f (0 or 1) so it correctly
    # tracks the per-step fall rate. Updating only on done=True would push it
    # permanently toward 1.0, breaking the curriculum gate.
    state.info["fall_rate_ema"] = (
        (1.0 - alpha) * state.info["fall_rate_ema"] + alpha * done_f
    )
    state.info["episode_step"] = jp.where(done, 0, state.info["episode_step"])
    state.info["push_step"] = jp.where(done, 0, state.info["push_step"])
    state.info["disturb_step"] = jp.where(done, 0, state.info["disturb_step"])
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["disturbance_profile_id"] = state.info["push_profile_id"].astype(
        jp.float32
    )

    return state.replace(
        data=data, obs=obs, reward=reward, done=done.astype(reward.dtype)
    )

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
  ) -> mjx_env.Observation:
    gyro = self.get_gyro(data, "pelvis")
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = gyro + (
        (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0, 0, -1])
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = gravity + (
        (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = joint_angles + (
        (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = joint_vel + (
        (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    zero_cmd = jp.zeros(3)
    zero_phase = jp.zeros(4)
    target_pose = info["target_pose"]
    # Keep semantics explicit: legs are target-relative, upper body is absolute.
    leg_residual = (
        noisy_joint_angles[self._leg_indices] - target_pose[self._leg_indices]
    )
    upper_abs = noisy_joint_angles[self._upper_indices]
    joint_state_obs = jp.hstack([leg_residual, upper_abs])
    state = jp.hstack([
        noisy_gyro,
        noisy_gravity,
        zero_cmd,
        joint_state_obs,
        noisy_joint_vel,
        info["last_act"],
        zero_phase,
    ])

    privileged_state = jp.hstack([
        state,
        gyro,
        self.get_accelerometer(data, "pelvis"),
        gravity,
        self.get_local_linvel(data, "pelvis"),
        self.get_global_angvel(data, "pelvis"),
        joint_angles - target_pose,
        joint_vel,
        data.qpos[2],
        data.actuator_force,
        contact,
    ])
    return {"state": state, "privileged_state": privileged_state}

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      done: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    qpos = data.qpos[7:]
    target_pose = info["target_pose"]
    torso_gravity = self.get_gravity(data, "torso")
    global_linvel = self.get_global_linvel(data, "pelvis")
    global_angvel = self.get_global_angvel(data, "torso")
    return {
        "orientation": self._cost_orientation(torso_gravity),
        "base_height": self._cost_base_height(data.qpos[2]),
        "lin_vel_z": jp.square(global_linvel[2]),
        "ang_vel_xy": jp.sum(jp.square(global_angvel[:2])),
        "base_linvel_xy": jp.sum(jp.square(global_linvel[:2])),
        "base_angvel_yaw": jp.square(global_angvel[2]),
        "com_stability": self._cost_com_stability(data, contact),
        # Only penalize leg joints — upper body is not controlled by the policy.
        "pose": jp.sum(jp.square(qpos[self._leg_indices] - target_pose[self._leg_indices])),
        "joint_vel": jp.sum(jp.square(data.qvel[6:18])),
        "dof_pos_limits": self._cost_joint_pos_limits(qpos),
        "action_rate": jp.sum(jp.square(action - info["last_act"])),
        "torques": jp.sum(jp.abs(data.actuator_force)),
        "energy": jp.sum(jp.abs(data.qvel[6:]) * jp.abs(data.actuator_force)),
        "dof_acc": jp.sum(jp.square(data.qacc[6:])),
        "collision": self._cost_collision(data),
        "contact_force": self._cost_contact_force(data),
        "foot_contact_symmetry": self._cost_foot_contact_symmetry(data, contact),
        "foot_flatness": self._cost_foot_flatness(data),
        "foot_slip": self._cost_foot_slip(data, contact, info["prev_foot_xy"]),
        "alive": jp.array(1.0),
        "still_bonus": self._reward_still_bonus(data),
        "termination": done,
    }

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall = self.get_gravity(data, "torso")[-1] < 0.0
    low_height = data.qpos[2] < 0.45
    collision = data.sensordata[
        self._mj_model.sensor_adr[self._right_foot_left_foot_found_sensor]
    ] > 0
    collision |= data.sensordata[
        self._mj_model.sensor_adr[self._left_foot_right_shin_found_sensor]
    ] > 0
    collision |= data.sensordata[
        self._mj_model.sensor_adr[self._right_foot_left_shin_found_sensor]
    ] > 0
    return fall | low_height | collision | jp.isnan(data.qpos).any() | jp.isnan(
        data.qvel
    ).any()

  def _get_foot_contact(self, data: mjx.Data) -> jax.Array:
    return jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._feet_floor_found_sensor
    ])

  def _sample_interval_steps(self, rng: jax.Array, interval_range) -> jax.Array:
    interval = jax.random.uniform(
      rng, minval=interval_range[0], maxval=interval_range[1]
    )
    return jp.round(interval / self.dt).astype(jp.int32)

  def _curriculum_progress(self, info: dict[str, Any]) -> jax.Array:
    if not self._config.curriculum_config.enable:
      return jp.array(1.0)
    cfg = self._config.curriculum_config
    time_progress = jp.clip(
        info["global_step"].astype(jp.float32) / cfg.ramp_steps, 0.0, 1.0
    )
    if not cfg.use_performance_gate:
      return time_progress
    survival_gate = jp.clip(
        info["episode_len_ema"] / cfg.target_episode_fraction, 0.0, 1.0
    )
    fall_gate = jp.clip(1.0 - info["fall_rate_ema"] / cfg.target_fall_rate, 0.0, 1.0)
    performance_progress = survival_gate * fall_gate
    return jp.minimum(time_progress, performance_progress)

  def _disturbances_enabled(self, info: dict[str, Any]) -> jax.Array:
    cfg = self._config.curriculum_config
    return info["global_step"] >= cfg.disturbance_warmup_steps

  def _lerp(self, start: float, end: float, alpha: jax.Array) -> jax.Array:
    return start + (end - start) * alpha

  def _sample_push(
      self, rng: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    progress = self._curriculum_progress(info)
    disturb_on = self._disturbances_enabled(info)
    push_mag_scale = self._lerp(
        self._config.curriculum_config.push_mag_scale_start,
        self._config.curriculum_config.push_mag_scale_end,
        progress,
    )
    rng, profile_rng, dir_rng, mag_rng = jax.random.split(rng, 4)
    profile = jax.random.randint(profile_rng, (), 0, 3)
    theta = jax.random.uniform(dir_rng, (), minval=0.0, maxval=2 * jp.pi)
    mag = jax.random.uniform(
        mag_rng,
        (),
        minval=self._config.push_config.magnitude_range[0] * push_mag_scale,
        maxval=self._config.push_config.magnitude_range[1] * push_mag_scale,
    )
    centered = jp.array([jp.cos(theta), jp.sin(theta)])
    lateral = jp.array([0.0, jp.sign(centered[1])])
    asymmetric = jp.array([centered[0], 0.35 * centered[1]])
    vec = jp.where(
        profile == 0,
        centered,
        jp.where(profile == 1, lateral, asymmetric),
    )
    do_push = (
        self._config.push_config.enable
        & disturb_on
        & (jp.mod(info["push_step"] + 1, info["push_interval_steps"]) == 0)
    )
    if self._config.curriculum_config.enable:
      interval_scale = self._lerp(
          self._config.curriculum_config.push_interval_scale_start,
          self._config.curriculum_config.push_interval_scale_end,
          progress,
      )
      rng, interval_rng = jax.random.split(rng)
      new_interval_steps = self._sample_interval_steps(
          interval_rng,
          [
              self._config.push_config.interval_range[0] * interval_scale,
              self._config.push_config.interval_range[1] * interval_scale,
          ],
      )
      info["push_interval_steps"] = jp.where(
          do_push, new_interval_steps, info["push_interval_steps"]
      )
    info["push_profile_id"] = profile
    return jp.where(do_push, vec * mag, jp.zeros(2))

  def _apply_joint_disturbance(
      self, targets: jax.Array, rng: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    if not self._config.joint_disturbance.enable:
      return targets
    disturb_on = self._disturbances_enabled(info)

    progress = self._curriculum_progress(info)
    disturb_mag_scale = self._lerp(
        self._config.curriculum_config.joint_mag_scale_start,
        self._config.curriculum_config.joint_mag_scale_end,
        progress,
    )
    disturb_interval_scale = self._lerp(
        self._config.curriculum_config.joint_interval_scale_start,
        self._config.curriculum_config.joint_interval_scale_end,
        progress,
    )
    start_new = (
        disturb_on
        &
        (info["disturb_time_left"] == 0)
        & (jp.mod(info["disturb_step"] + 1, info["disturb_interval_steps"]) == 0)
    )
    rng, joint_rng, mag_rng, dur_rng, sign_rng, int_rng = jax.random.split(rng, 6)
    new_joint = jax.random.randint(joint_rng, (), 0, 12)
    new_mag = jax.random.uniform(
        mag_rng,
        (),
        minval=self._config.joint_disturbance.magnitude_range[0]
        * disturb_mag_scale,
        maxval=self._config.joint_disturbance.magnitude_range[1]
        * disturb_mag_scale,
    )
    new_dur = jax.random.randint(
        dur_rng,
        (),
        self._config.joint_disturbance.duration_range[0],
        self._config.joint_disturbance.duration_range[1] + 1,
    )
    new_sign = jp.where(jax.random.bernoulli(sign_rng, 0.5), 1.0, -1.0)
    new_interval = self._sample_interval_steps(
        int_rng,
        [
            self._config.joint_disturbance.interval_range[0]
            * disturb_interval_scale,
            self._config.joint_disturbance.interval_range[1]
            * disturb_interval_scale,
        ],
    )

    info["disturb_joint"] = jp.where(start_new, new_joint, info["disturb_joint"])
    info["disturb_value"] = jp.where(
        start_new, new_sign * new_mag, info["disturb_value"]
    )
    info["disturb_time_left"] = jp.where(start_new, new_dur, info["disturb_time_left"])
    info["disturb_interval_steps"] = jp.where(
        start_new, new_interval, info["disturb_interval_steps"]
    )

    active = disturb_on & (info["disturb_time_left"] > 0)
    index = info["disturb_joint"]
    disturbed = targets.at[index].add(jp.where(active, info["disturb_value"], 0.0))
    return disturbed

  def _sample_upper_body_episode_offset(
      self, rng: jax.Array, info: dict[str, Any]
  ) -> tuple[jax.Array, jax.Array]:
    """Sample one upper joint index and signed offset (rad) for the next episode."""
    progress = self._curriculum_progress(info)
    disturb_mag_scale = self._lerp(
        self._config.curriculum_config.upper_mag_scale_start,
        self._config.curriculum_config.upper_mag_scale_end,
        progress,
    )
    rng, joint_rng, mag_rng, sign_rng = jax.random.split(rng, 4)
    new_joint = jax.random.randint(
        joint_rng, (), 0, int(self._upper_indices.shape[0])
    )
    cfg = self._config.upper_body_pose_disturbance
    new_mag = jax.random.uniform(
        mag_rng,
        (),
        minval=cfg.magnitude_range[0] * disturb_mag_scale,
        maxval=cfg.magnitude_range[1] * disturb_mag_scale,
    )
    new_sign = jp.where(jax.random.bernoulli(sign_rng, 0.5), 1.0, -1.0)
    raw = new_sign * new_mag
    lo = self._upper_offset_min[new_joint]
    hi = self._upper_offset_max[new_joint]
    clamped = jp.clip(raw, lo, hi)
    return new_joint, clamped

  def _apply_upper_body_episode_offset(
      self, targets: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    if not self._config.upper_body_pose_disturbance.enable:
      return targets
    disturb_on = self._disturbances_enabled(info)
    upper_index = self._upper_indices[info["upper_episode_joint"]]
    return targets.at[upper_index].add(
        jp.where(disturb_on, info["upper_episode_offset"], 0.0)
    )

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torso_zaxis - jp.array([0.0, 0.0, 1.0])))

  def _cost_base_height(self, base_height: jax.Array) -> jax.Array:
    return jp.square(base_height - self._config.reward_config.base_height_target)

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _cost_collision(self, data: mjx.Data) -> jax.Array:
    c = (
        data.sensordata[
            self._mj_model.sensor_adr[self._left_hand_left_thigh_found_sensor]
        ]
        > 0
    )
    c |= (
        data.sensordata[
            self._mj_model.sensor_adr[self._right_hand_right_thigh_found_sensor]
        ]
        > 0
    )
    return jp.any(c)

  def _cost_contact_force(self, data: mjx.Data) -> jax.Array:
    l_force = mjx_env.get_sensor_data(self.mj_model, data, "left_foot_force")[2]
    r_force = mjx_env.get_sensor_data(self.mj_model, data, "right_foot_force")[2]
    max_force = self._config.reward_config.max_contact_force
    cost = jp.clip(jp.abs(l_force) - max_force, min=0.0)
    cost += jp.clip(jp.abs(r_force) - max_force, min=0.0)
    return cost

  def _cost_foot_contact_symmetry(
      self, data: mjx.Data, contact: jax.Array
  ) -> jax.Array:
    l_force = jp.abs(mjx_env.get_sensor_data(self.mj_model, data, "left_foot_force")[2])
    r_force = jp.abs(mjx_env.get_sensor_data(self.mj_model, data, "right_foot_force")[2])
    balance = jp.abs(l_force - r_force) / (l_force + r_force + 1e-6)
    single_support = jp.logical_xor(contact[0], contact[1]).astype(jp.float32)
    return balance + single_support

  def _cost_foot_flatness(self, data: mjx.Data) -> jax.Array:
    l_up = data.sensordata[self._left_foot_upvector_adr : self._left_foot_upvector_adr + 3]
    r_up = data.sensordata[
        self._right_foot_upvector_adr : self._right_foot_upvector_adr + 3
    ]
    target = jp.array([0.0, 0.0, 1.0])
    return jp.sum(jp.square(l_up - target)) + jp.sum(jp.square(r_up - target))

  def _cost_foot_slip(
      self, data: mjx.Data, contact: jax.Array, prev_foot_xy: jax.Array
  ) -> jax.Array:
    # Tangential (ground-plane) slip velocity of feet, penalized only in contact.
    curr_foot_xy = data.site_xpos[self._feet_site_id, :2]
    foot_vel_xy = (curr_foot_xy - prev_foot_xy) / self.dt
    foot_speed_sq = jp.sum(jp.square(foot_vel_xy), axis=1)
    contact_f = contact.astype(foot_speed_sq.dtype)
    contact_count = jp.maximum(jp.sum(contact_f), 1.0)
    return jp.sum(foot_speed_sq * contact_f) / contact_count

  def _reward_still_bonus(self, data: mjx.Data) -> jax.Array:
    lin_xy = self.get_global_linvel(data, "pelvis")[:2]
    yaw = self.get_global_angvel(data, "torso")[2]
    tilt = jp.linalg.norm(self.get_gravity(data, "torso")[:2])
    err = jp.linalg.norm(lin_xy) + jp.abs(yaw) + tilt
    # Softer exponent gives meaningful gradient even when robot is wobbly.
    return jp.exp(-1.5 * err)

  def _cost_com_stability(self, data: mjx.Data, contact: jax.Array) -> jax.Array:
    com_xy = data.subtree_com[self._pelvis_body_id, :2]  # whole-body CoM
    foot_center_xy = jp.mean(data.site_xpos[self._feet_site_id, :2], axis=0)
    cost = jp.sum(jp.square(com_xy - foot_center_xy))
    both_feet_contact = jp.all(contact).astype(jp.float32)
    return cost * (0.25 + 0.75 * both_feet_contact)
