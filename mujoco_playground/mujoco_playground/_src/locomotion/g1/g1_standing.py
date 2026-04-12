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
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import base as g1_base
from mujoco_playground._src.locomotion.g1 import g1_constants as consts


# Library of standing poses: each row is [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll] × 2 (L then R).
# Use num_poses in the config to select how many poses (from index 0 upward) are sampled during reset.
LEG_POSE_LIBRARY = np.array(
    [
        # fmt: off
        [-0.20,  0.00,  0.00, 0.59, -0.34,  0.00, -0.20,  0.00,  0.00, 0.59, -0.34,  0.00],  # h≈0.728
        [-0.20,  0.10,  0.08, 0.59, -0.34,  0.05, -0.20, -0.10, -0.08, 0.59, -0.34, -0.05],  # h≈0.727
        [-0.22,  0.20,  0.10, 0.65, -0.38,  0.08, -0.22, -0.20, -0.10, 0.65, -0.38, -0.08],  # h≈0.717
        [-0.25,  0.32,  0.14, 0.80, -0.48,  0.13, -0.25, -0.32, -0.14, 0.80, -0.48, -0.13],  # h≈0.691
        [-0.10,  0.00,  0.00, 0.25, -0.14,  0.00, -0.10,  0.00,  0.00, 0.25, -0.14,  0.00],  # h≈0.752
        [-0.10,  0.12,  0.06, 0.25, -0.14,  0.05, -0.10, -0.12, -0.06, 0.25, -0.14, -0.05],  # h≈0.749
        [-0.20,  0.15,  0.20, 0.59, -0.34,  0.06, -0.20, -0.15, -0.20, 0.59, -0.34, -0.06],  # h≈0.728
        [-0.22,  0.25,  0.10, 0.78, -0.46,  0.10, -0.18, -0.08, -0.04, 0.45, -0.26, -0.03],  # h≈0.717
        [-0.25,  0.08,  0.06, 0.85, -0.50,  0.03, -0.12, -0.15, -0.08, 0.30, -0.17, -0.06],  # h≈0.720
        [-0.12,  0.35,  0.10, 0.30, -0.17,  0.14, -0.12, -0.35, -0.10, 0.30, -0.17, -0.14],  # h≈0.722
        [-0.18,  0.12,  0.10, 0.55, -0.32,  0.05, -0.18, -0.12, -0.10, 0.55, -0.32, -0.05],  # h≈0.730
        [-0.22,  0.28,  0.12, 0.72, -0.43,  0.11, -0.22, -0.28, -0.12, 0.72, -0.43, -0.11],  # h≈0.702
        # fmt: on
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
      # Number of poses to sample from LEG_POSE_LIBRARY (uses poses [0 .. num_poses-1]).
      num_poses=10,
      # Pelvis orientation [w, x, y, z] — identity = upright (0° roll/pitch/yaw).
      base_quat=[1.0, 0.0, 0.0, 0.0],
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
              base_angvel_yaw=-0.1,
              com_stability=-0.8,
              pose=-0.5,
              joint_vel=-0.01,
              dof_pos_limits=-1.0,
              action_rate=-0.03,
              torques=0.0,
              energy=0.0,
              dof_acc=0.0,
              collision=-2.0,
              contact_force=-0.01,
              foot_contact_symmetry=-0.2,
              foot_flatness=-0.2,
              foot_distance=-15.0,
              foot_slip=-0.5,
              alive=1.0,
              still_bonus=1.0,
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
      curriculum_config=config_dict.create(
          enable=True,
          ramp_steps=10_000_000,
          disturbance_warmup_steps=1_000_000,
          use_performance_gate=True,
          ema_alpha=0.02,
          target_episode_fraction=0.8,
          target_fall_rate=0.2,
          push_mag_scale_start=0.25,
          push_mag_scale_end=1.0,
          push_interval_scale_start=2.0,  # larger => less frequent early
          push_interval_scale_end=1.0,
          joint_mag_scale_start=0.25,
          joint_mag_scale_end=1.0,
          joint_interval_scale_start=2.0,
          joint_interval_scale_end=1.0,
      ),
      support_height_perturb_range=[-0.01, 0.02],
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

    n = int(self._config.num_poses)
    assert 1 <= n <= len(LEG_POSE_LIBRARY), f"num_poses must be in [1, {len(LEG_POSE_LIBRARY)}]"
    self._leg_pose_library = jp.array(LEG_POSE_LIBRARY[:n])  # (n, 12)
    self._base_quat = jp.array(self._config.base_quat)
    self._upper_target = jp.array(self._config.upper_body_target)

    # PD gains for leg joints.
    # - kp from leg position actuators (actuators 0-11).
    # - kd from leg joint damping (DOFs 6-17 in qvel/qfrc_bias space).
    # We use both for feedforward-to-position conversion.
    self._leg_kp = jp.array(self._mj_model.actuator_gainprm[:12, 0])
    self._leg_kd = jp.array(self._mj_model.dof_damping[6:18])

    self._leg_indices = jp.arange(12)
    self._upper_indices = jp.arange(12, 29)
    self._hip_indices = jp.array([1, 2, 7, 8])
    self._knee_indices = jp.array([3, 9])

    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

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

    # Body IDs for the feet — used to read cvel for slip penalty.
    self._feet_body_id = np.array(
        [self._mj_model.site_bodyid[sid] for sid in self._feet_site_id]
    )

    # Per-pose target foot distance (XY plane) computed via CPU forward pass.
    init_qpos = np.array(self._mj_model.keyframe("knees_bent").qpos)
    foot_distances = []
    for pose in LEG_POSE_LIBRARY[:n]:
      d = mujoco.MjData(self._mj_model)
      d.qpos[:] = init_qpos
      d.qpos[7:19] = pose
      mujoco.mj_forward(self._mj_model, d)
      l_xy = d.site_xpos[self._feet_site_id[0], :2]
      r_xy = d.site_xpos[self._feet_site_id[1], :2]
      foot_distances.append(float(np.linalg.norm(l_xy - r_xy)))
    self._foot_target_distances = jp.array(foot_distances)  # (n,)

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
    pose_idx = jax.random.randint(key, (), 0, len(self._leg_pose_library))
    leg_pose = self._leg_pose_library[pose_idx]

    target_pose = self._default_pose
    target_pose = target_pose.at[self._leg_indices].set(leg_pose)
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

    info = {
        "rng": rng,
        "step": jp.array(0, dtype=jp.int32),
        "global_step": jp.array(0, dtype=jp.int32),
        "episode_step": jp.array(0, dtype=jp.int32),
        "episode_len_ema": jp.array(0.0),
        "fall_rate_ema": jp.array(1.0),
        "target_pose": target_pose,
        "last_act": jp.zeros(self.action_size),
        "last_last_act": jp.zeros(self.action_size),
        "motor_targets": target_pose,
        "push_step": jp.array(0, dtype=jp.int32),
        "push_interval_steps": push_interval_steps,
        "disturb_step": jp.array(0, dtype=jp.int32),
        "disturb_interval_steps": disturb_interval_steps,
        "disturb_time_left": jp.array(0, dtype=jp.int32),
        "disturb_joint": jp.array(0, dtype=jp.int32),
        "disturb_value": jp.array(0.0),
        "push_profile_id": jp.array(0, dtype=jp.int32),
        "pose_idx": pose_idx,
    }

    metrics = {
        f"reward/{k}": jp.zeros(())
        for k in self._config.reward_config.scales.keys()
    }
    metrics["disturbance_profile_id"] = jp.zeros(())

    contact = self._get_foot_contact(data)
    obs = self._get_obs(data, info, contact)
    return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    rng = state.info["rng"]
    rng, push_rng, disturb_rng = jax.random.split(rng, 3)

    data = state.data
    push = self._sample_push(push_rng, state.info)
    qvel = data.qvel.at[:2].set(data.qvel[:2] + push)
    data = data.replace(qvel=qvel)

    leg_targets = (
        state.info["target_pose"][self._leg_indices]
        + action * self._config.action_scale
    )
    motor_targets = state.info["target_pose"].at[self._leg_indices].set(leg_targets)
    motor_targets = motor_targets.at[self._upper_indices].set(self._upper_target)

    motor_targets = self._apply_joint_disturbance(
        motor_targets, disturb_rng, state.info
    )
    motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)

    data = mjx_env.step(self.mjx_model, data, motor_targets, self.n_substeps)
    contact = self._get_foot_contact(data)
    done = self._get_termination(data)

    obs = self._get_obs(data, state.info, contact)
    rewards = self._get_reward(data, action, state.info, done, contact)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = sum(rewards.values()) * self.dt

    state.info["rng"] = rng
    state.info["step"] += 1
    state.info["global_step"] += 1
    state.info["episode_step"] += 1
    state.info["push_step"] += 1
    state.info["disturb_step"] += 1
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["motor_targets"] = motor_targets

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

    quat = data.qpos[3:7]
    quat = quat / (jp.linalg.norm(quat) + 1e-8)
    qw, qx, qy, qz = quat
    gravity = jp.array([
        2.0 * (-qz * qx + qw * qy),
        -2.0 * (qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz),
    ])
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
        "foot_distance": self._cost_foot_distance(data, info),
        "foot_slip": self._cost_foot_slip(data, contact),
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

  def _cost_foot_distance(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    """Penalise deviation from the per-pose target XY foot separation."""
    l_xy = data.site_xpos[self._feet_site_id[0], :2]
    r_xy = data.site_xpos[self._feet_site_id[1], :2]
    dist = jp.linalg.norm(l_xy - r_xy)
    target = self._foot_target_distances[info["pose_idx"]]
    return jp.square(dist - target)

  def _cost_foot_slip(self, data: mjx.Data, contact: jax.Array) -> jax.Array:
    """Penalise XY foot velocity when the foot is in contact with the ground."""
    # data.cvel shape: (nbody, 6) — [angvel(3), linvel(3)] in world frame.
    l_linvel_xy = data.cvel[self._feet_body_id[0], 3:5]
    r_linvel_xy = data.cvel[self._feet_body_id[1], 3:5]
    l_slip = jp.sum(jp.square(l_linvel_xy)) * contact[0]
    r_slip = jp.sum(jp.square(r_linvel_xy)) * contact[1]
    return l_slip + r_slip

  def _reward_still_bonus(self, data: mjx.Data) -> jax.Array:
    lin_xy = self.get_global_linvel(data, "pelvis")[:2]
    yaw = self.get_global_angvel(data, "torso")[2]
    tilt = jp.linalg.norm(self.get_gravity(data, "torso")[:2])
    err = jp.linalg.norm(lin_xy) + jp.abs(yaw) + tilt
    # Softer exponent gives meaningful gradient even when robot is wobbly.
    return jp.exp(-1.5 * err)

  def _cost_com_stability(self, data: mjx.Data, contact: jax.Array) -> jax.Array:
    com_xy = data.subtree_com[self._torso_body_id, :2]
    foot_center_xy = jp.mean(data.site_xpos[self._feet_site_id, :2], axis=0)
    cost = jp.sum(jp.square(com_xy - foot_center_xy))
    both_feet_contact = jp.all(contact).astype(jp.float32)
    return cost * (0.25 + 0.75 * both_feet_contact)
