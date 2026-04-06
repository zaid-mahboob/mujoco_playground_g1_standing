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
"""Utilities for randomization."""
import jax
import jax.numpy as jp
from mujoco import mjx

FLOOR_GEOM_ID = 0
TORSO_BODY_ID = 16


def domain_randomize(model: mjx.Model, rng: jax.Array):
  @jax.vmap
  def rand_dynamics(rng):
    # Floor / foot friction: =U(0.3, 1.2).
    rng, key = jax.random.split(rng)
    friction = jax.random.uniform(key, minval=0.3, maxval=1.2)
    pair_friction = model.pair_friction.at[0:2, 0:2].set(friction)

    # Scale static friction: *U(0.4, 2.4).
    rng, key = jax.random.split(rng)
    frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
        key, shape=(29,), minval=0.4, maxval=2.4
    )
    dof_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

    # Scale dof damping: *U(0.7, 1.3).
    rng, key = jax.random.split(rng)
    damping = model.dof_damping[6:] * jax.random.uniform(
        key, shape=(29,), minval=0.7, maxval=1.3
    )
    dof_damping = model.dof_damping.at[6:].set(damping)

    # Scale armature: *U(0.95, 1.10).
    rng, key = jax.random.split(rng)
    armature = model.dof_armature[6:] * jax.random.uniform(
        key, shape=(29,), minval=0.95, maxval=1.10
    )
    dof_armature = model.dof_armature.at[6:].set(armature)

    # Scale body inertias: *U(0.8, 1.2).
    rng, key = jax.random.split(rng)
    inertia_scale = jax.random.uniform(
        key, shape=model.body_inertia.shape, minval=0.8, maxval=1.2
    )
    body_inertia = model.body_inertia.at[:].set(model.body_inertia * inertia_scale)

    # Scale all link masses: *U(0.8, 1.2).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(
        key, shape=(model.nbody,), minval=0.8, maxval=1.2
    )
    body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

    # Add mass to torso: +U(-1.5, 1.5).
    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(key, minval=-1.5, maxval=1.5)
    body_mass = body_mass.at[TORSO_BODY_ID].set(
        body_mass[TORSO_BODY_ID] + dmass
    )

    # Gravity direction tilt: roll/pitch U(-3deg, 3deg), keep magnitude.
    rng, key_r, key_p = jax.random.split(rng, 3)
    max_tilt = jp.deg2rad(3.0)
    roll = jax.random.uniform(key_r, minval=-max_tilt, maxval=max_tilt)
    pitch = jax.random.uniform(key_p, minval=-max_tilt, maxval=max_tilt)
    gmag = jp.linalg.norm(model.opt.gravity)
    gravity = jp.array(
        [
            gmag * jp.sin(pitch),
            -gmag * jp.sin(roll),
            -gmag * jp.cos(pitch) * jp.cos(roll),
        ],
        dtype=model.opt.gravity.dtype,
    )

    # Jitter qpos0: +U(-0.10, 0.10).
    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0
    qpos0 = qpos0.at[7:].set(
        qpos0[7:]
        + jax.random.uniform(key, shape=(29,), minval=-0.10, maxval=0.10)
    )

    return (
        pair_friction,
        dof_frictionloss,
        dof_damping,
        dof_armature,
        body_inertia,
        body_mass,
        gravity,
        qpos0,
    )

  (
      pair_friction,
      frictionloss,
      dof_damping,
      armature,
      body_inertia,
      body_mass,
      gravity,
      qpos0,
  ) = rand_dynamics(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "pair_friction": 0,
      "dof_frictionloss": 0,
      "dof_damping": 0,
      "dof_armature": 0,
      "body_inertia": 0,
      "body_mass": 0,
      "opt.gravity": 0,
      "qpos0": 0,
  })

  model = model.tree_replace({
      "pair_friction": pair_friction,
      "dof_frictionloss": frictionloss,
      "dof_damping": dof_damping,
      "dof_armature": armature,
      "body_inertia": body_inertia,
      "body_mass": body_mass,
      "opt.gravity": gravity,
      "qpos0": qpos0,
  })

  return model, in_axes
