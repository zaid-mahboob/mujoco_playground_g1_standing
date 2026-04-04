# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU-accelerated RL environment and training framework for a standing policy on the Unitree G1 humanoid robot, built on Google DeepMind's MuJoCo Playground (MJX). Training uses RSL-RL (Legged Robotics Lab) PPO.

## Key Commands

All commands run from `mujoco_playground/` unless otherwise specified.

**Training (RSL-RL PPO):**
```bash
python -m learning.train_rsl_rl --env_name=G1StandingFlatTerrain --num_envs=4096 --device=cuda:0
# Resume from checkpoint:
python -m learning.train_rsl_rl --env_name=G1StandingFlatTerrain --load_run_name <run_name> --checkpoint_num <num>
# Inference + video render:
python -m learning.train_rsl_rl --env_name=G1StandingFlatTerrain --play_only --load_run_name <run_name>
```

**Training (JAX/Brax PPO alternative):**
```bash
python -m learning.train_jax_ppo --env_name=G1StandingFlatTerrain
```

**Utility scripts:**
```bash
cd mujoco_playground/scripts
python view_g1_standing_pose.py        # Visualize pose + COM/support ratio
python test_standing_pose.py           # PD controller feasibility test
python test_standing_pose.py --no_viewer --duration 10
```

**Tests:**
```bash
pytest mujoco_playground/_src/locomotion/locomotion_test.py -xvs
```

**Linting/formatting:**
```bash
ruff check . && ruff format .
```

## Architecture

### Environment Layer
`mujoco_playground/_src/locomotion/g1/g1_standing.py` — The core RL environment.

- **Action space (12D):** Residual offsets to leg joint targets (6L + 6R). Upper body targets are fixed to zero.
- **Observation space:** Two separate obs tensors — `state` (policy obs: target-relative leg angles, absolute upper-body angles, gyro, gravity, joint vel) and `privileged_state` (critic obs: richer). Linear velocity is omitted for sim-to-real alignment.
- **Reset:** Samples from two standing templates with quaternion-specific pelvis orientation. Adds base XY randomization + support-height perturbation.
- **Reward:** 10+ terms covering orientation, base height, velocity, COM stability, leg pose tracking, joint limits, action smoothness, foot flatness/symmetry, alive bonus.
- **Disturbances:** Push disturbances (velocity kicks) + joint pulses (target offsets), both curriculum-gated.
- **Curriculum:** Time-ramped + performance-gated via episode survival and fall-rate EMAs. 1M-step warmup before disturbances engage.
- **Feedforward compensation:** Gravity/Coriolis torques from `data.qfrc_bias` converted to position offsets via PD gains, keeping RL output as residuals.

### Training Layer
`mujoco_playground/learning/train_rsl_rl.py` — PPO training with RSL-RL.

- `RSLRLBraxWrapper` converts MJX (JAX) environments to PyTorch for RSL-RL's `OnPolicyRunner`.
- Actor and critic both use [512, 256, 128] hidden layers with empirical normalization.
- Policy observes `state`; critic observes `privileged_state`.
- Outputs to `runs/<exp_name>/checkpoints/` (`.pt` files) + tensorboard logs.

### RL Config Layer
`mujoco_playground/mujoco_playground/config/locomotion_params.py` — RL hyperparameters per env.

- G1Standing: 120M timesteps, entropy_cost=0.002, clipping_epsilon=0.2.
- Base PPO: lr=3e-4, gamma=0.99, lam=0.95, 24 steps/env, save every 50 iterations.

### Environment Registry
`mujoco_playground/_src/locomotion/__init__.py` registers:
- `"G1StandingFlatTerrain"` and `"G1StandingRoughTerrain"` → `g1_standing.Standing`
- Domain randomizer: `g1_randomize.domain_randomize`

### Deployment Config
`configs/g1.yaml` — Hardware deployment settings: 50 Hz control (decimation=10 at sim_dt=0.002), 12 PD joints, observation/action scales.

## Key Design Notes

- The environment is designed for **sim-to-real transfer**: pelvis linear velocity is excluded from observations, and gravity feedforward compensation is baked in.
- `G1_STANDING_RL_CONFIGURATION.md` inside `mujoco_playground/` is the authoritative design document for all reward weights, curriculum thresholds, and disturbance parameters.
- Pre-trained model checkpoint is at `model/model_7750.pt`.
- The `.venv/` inside `mujoco_playground/` is the active Python environment (Python 3.12, JAX CUDA 12).
- On Ampere+ GPUs, TF32 may cause reproducibility issues — see the project README FAQ.
