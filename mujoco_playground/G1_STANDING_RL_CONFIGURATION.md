# G1 Standing Controller RL Configuration

This document explains how the current G1 standing controller environment is configured in this repository, in plain language, for handoff/review.

## Goal

Train a robust **standing** policy for Unitree G1 where:

- the policy controls **legs only**,
- upper body is controlled internally by fixed targets,
- the robot can stand under disturbances and randomization.

---

## Where the setup is implemented

- Environment: `mujoco_playground/_src/locomotion/g1/g1_standing.py`
- Environment registration: `mujoco_playground/_src/locomotion/__init__.py`
- RL config branch: `mujoco_playground/config/locomotion_params.py`

Registered environment names:

- `G1StandingFlatTerrain`
- `G1StandingRoughTerrain`

---

## Action space (policy output)

The standing policy outputs **12 actions** (legs only).

- Left leg: 6 joints
- Right leg: 6 joints

In the env:

1. A per-episode `target_pose` is sampled at reset.
2. Policy output is applied only on the 12 leg joints:
   - `leg_target = target_leg + action * action_scale`
3. Upper body targets are overwritten to fixed upper-body values.
4. Full 29-dim motor target is sent to MuJoCo position actuators.

So upper body is **not** directly controlled by the policy.

---

## Observation space (policy input)

Policy observation (`obs["state"]`) is target-relative and does **not** include pelvis linear velocity.

It contains:

- noisy gyro
- noisy gravity
- zero command placeholder
- joint state split explicitly:
  - leg joints: residual to leg target pose
  - upper-body joints: absolute/noisy joint angles
- noisy joint velocities
- last action
- zero phase placeholder

Important design choices:

- Observation reference is explicit by body part:
  - legs are target-relative
  - upper body is absolute
- Pelvis linear velocity was removed from actor input for better real-robot alignment.

Critic observation (`obs["privileged_state"]`) is richer and still includes extra simulator signals (including local linear velocity).

---

## Reset behavior

At each reset:

1. Base starts from `knees_bent` keyframe.
2. One of two standing templates is sampled:
   - `leg_pose_a = [-0.165, 0.4927, 0.1239, 0.8178, -0.528, -0.26, -0.125, -0.622, -0.8648, 0.89, -0.263, 0.137]`
   - `leg_pose_b = [-0.125, 0.622, 0.8648, 0.89, -0.263, -0.137, -0.165, -0.4927, -0.1239, 0.8178, -0.528, 0.26]`
3. Pose-specific pelvis orientation is applied:
   - `leg_pose_a_quat = [0.9990482, -0.0436194, 0.0, 0.0]`
   - `leg_pose_b_quat = [0.9990482, 0.0436194, 0.0, 0.0]`
4. Small random yaw is applied on top of that base quaternion.
5. Base XY randomization and support-height perturbation are applied.
6. Upper body target is fixed (zeros).
7. Small joint target noise is applied and clipped to limits.
8. `target_pose` is stored in `state.info` for action/reward/obs use.

---

## Reward design (standing-focused)

Main active terms:

- orientation
- base height
- base velocity penalties (vertical, XY drift, yaw rate)
- COM stability (`com_stability`)
- pose tracking to per-episode `target_pose` **for leg joints only**
- joint limit penalty
- action rate smoothness
- contact/collision quality
- foot contact symmetry
- foot flatness
- alive bonus
- still bonus (with softer shaping for better early gradient)
- termination penalty

Removed/reduced locomotion-style terms:

- gait/command tracking style rewards are not used for standing.
- hip/knee deviation terms were removed as redundant after pose cleanup.

---

## Disturbances and curriculum

Disturbances implemented:

1. **Push disturbances** (with multiple profile modes)
2. **Joint disturbance pulses** (target perturbations on selected leg joints)
3. **Support-height perturbation** at reset

Disturbance details and units:

| Disturbance | Where applied | Config range | Units | Notes |
|---|---|---|---|---|
| Push | `qvel[:2]` (base XY velocity) | `push_config.magnitude_range = [0.05, 1.0]` | m/s | Added as velocity kick in XY plane |
| Push interval | Scheduler | `push_config.interval_range = [3.0, 7.0]` | s | Time between push events |
| Joint pulse | `motor_targets[joint] += delta` | `joint_disturbance.magnitude_range = [0.02, 0.12]` | rad | Temporary target offset on one leg joint |
| Joint pulse interval | Scheduler | `joint_disturbance.interval_range = [1.5, 4.0]` | s | Time between pulse events |
| Joint pulse duration | Scheduler | `joint_disturbance.duration_range = [2, 8]` | steps | With `dt=0.02`, this is `0.04-0.16 s` |
| Support height perturb | Reset `qpos[2]` | `support_height_perturb_range = [-0.01, 0.02]` | m | Per-episode initial base-height offset |

What "interval" means:

- Interval is the sampled waiting time before the next disturbance event.
- Larger interval => disturbances happen less frequently.
- Smaller interval => disturbances happen more frequently.

Curriculum is time-ramped but **performance-gated**:

- time component: linear ramp over `global_step`
- gate component: episode survival/fall EMAs
  - `episode_len_ema` (longer survival unlocks harder disturbances)
  - `fall_rate_ema` (lower falls unlock harder disturbances, updated every step)

Curriculum progress uses:

- `progress = min(time_progress, performance_progress)`
- `performance_progress = survival_gate * fall_gate`

This is controlled via `curriculum_config` in `default_config()`.

Curriculum effect on disturbance strength/frequency:

| Parameter | Early training (`progress≈0`) | Late training (`progress≈1`) | Effect |
|---|---:|---:|---|
| Push magnitude scale | `0.25` | `1.0` | Pushes get stronger |
| Push interval scale | `2.0` | `1.0` | Pushes get more frequent over time |
| Joint magnitude scale | `0.25` | `1.0` | Joint pulses get stronger |
| Joint interval scale | `2.0` | `1.0` | Joint pulses get more frequent over time |

Disturbance warmup:

- `curriculum_config.disturbance_warmup_steps = 100000`
- Before this step count, push and joint pulse disturbances are disabled.
- This gives the policy a clean standing phase before robustness stress is introduced.

---

## Feedforward compensation (GC + damping)

Leg targets include feedforward compensation so RL does not need to learn basic gravity/Coriolis cancellation from scratch.

At each step:

1. Read bias forces for leg DOFs from `data.qfrc_bias[6:18]`.
2. Read leg PD gains:
   - `kp` from actuator gains (`actuator_gainprm[:12, 0]`)
   - `kd` from joint damping (`dof_damping[6:18]`)
3. Convert desired feedforward torque to position-target offset:
   - `delta_target = (qfrc_bias + kd * qvel_leg) / kp`
4. Add `delta_target` to leg motor targets (clipped for safety).

This keeps policy output as a residual around a physically meaningful baseline.

---

## Domain randomization

Both standing envs are wired to existing G1 domain randomization:

- `g1_randomize.domain_randomize`

through locomotion registry mapping.

---

## RL hyperparameter wiring

`locomotion_params.py` includes a dedicated RSL-RL branch for:

- `G1StandingFlatTerrain`
- `G1StandingRoughTerrain`

and includes them in the shorter-iteration bring-up list.

---

## Current status summary

The implementation is internally consistent on key points:

- policy action is leg-only (12D),
- action, reward, and observation all use per-episode `target_pose`,
- two deployment standing templates + pose-specific pelvis quaternions are used at reset,
- pelvis height target is set to `0.715`,
- gravity/Coriolis+damping feedforward compensation is active on leg targets,
- standing env is registered and trainable through the standard training entrypoint,
- disturbance curriculum is present,
- lints are clean for changed files.

---

## Training command example

`python -m learning.train_rsl_rl --env_name=G1StandingFlatTerrain --num_envs=4096 --device=cuda:0`

