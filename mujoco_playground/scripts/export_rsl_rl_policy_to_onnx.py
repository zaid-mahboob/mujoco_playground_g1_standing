#!/usr/bin/env python3
"""Export an RSL-RL ActorCritic checkpoint (.pt) to ONNX for onnxruntime.

Matches inference path: actor observation normalization + actor MLP mean
(same as ``policy.act_inference({"state": obs})`` for a single ``state`` group).

Example (``mujoco_playground`` venv active)::

  cd mujoco_playground
  python scripts/export_rsl_rl_policy_to_onnx.py \\
    --checkpoint ../model/model_7750.pt \\
    --output ../model/g1_standing_policy.onnx

Install ``onnxruntime`` (and ``numpy``) to run the post-export numerical check; use ``--no-verify`` to skip it.

Default obs layout for G1 standing is 83-D ``state`` (see ``g1_standing._get_obs``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

# rsl_rl must match the environment used for training.
from rsl_rl.modules import ActorCritic


def _infer_dims(state_dict: dict[str, Any]) -> tuple[int, int, int]:
  """Return (num_actor_obs, num_actions, num_critic_obs) from weights."""
  n_act = int(state_dict["actor.6.weight"].shape[0])
  n_obs = int(state_dict["actor.0.weight"].shape[1])
  # Critic last linear layer key may vary if hidden_dims differ; use first layer.
  critic_w = state_dict["critic.0.weight"]
  n_crit = int(critic_w.shape[1])
  return n_obs, n_act, n_crit


def _infer_hidden_dims(state_dict: dict[str, Any], prefix: str) -> list[int]:
  """Infer MLP hidden sizes from Linear layers (all even indices until last)."""
  i = 0
  hidden: list[int] = []
  while True:
    wkey = f"{prefix}.{i}.weight"
    if wkey not in state_dict:
      break
    out_d, in_d = state_dict[wkey].shape
    # Last linear maps to actions (actor) or value (critic)
    nxt = f"{prefix}.{i + 2}.weight"
    if nxt not in state_dict:
      break
    hidden.append(out_d)
    i += 2
  return hidden


def _has_actor_norm(state_dict: dict[str, Any]) -> bool:
  return "actor_obs_normalizer._mean" in state_dict


def _build_actor_critic(
    state_dict: dict[str, Any],
    obs_groups: dict[str, list[str]],
    policy_extras: dict[str, Any] | None,
) -> ActorCritic:
  n_obs, n_act, n_crit = _infer_dims(state_dict)
  actor_hidden = _infer_hidden_dims(state_dict, "actor")
  critic_hidden = _infer_hidden_dims(state_dict, "critic")
  if not actor_hidden or not critic_hidden:
    raise ValueError("Could not infer hidden dims from checkpoint.")

  obs = TensorDict(
      {
          "state": torch.zeros(1, n_obs),
          "privileged_state": torch.zeros(1, n_crit),
      },
      batch_size=[1],
  )

  actor_norm = _has_actor_norm(state_dict)
  critic_norm = "critic_obs_normalizer._mean" in state_dict

  kwargs: dict[str, Any] = {
      "actor_hidden_dims": actor_hidden,
      "critic_hidden_dims": critic_hidden,
      "activation": "elu",
      "init_noise_std": 1.0,
      "actor_obs_normalization": actor_norm,
      "critic_obs_normalization": critic_norm,
  }
  if policy_extras:
    kwargs.update(policy_extras)

  policy = ActorCritic(obs, obs_groups, n_act, **kwargs)
  policy.load_state_dict(state_dict, strict=True)
  policy.eval()
  return policy


class ActorInferenceWrapper(nn.Module):
  """ONNX-friendly: normalized obs -> action mean (no sampling, no TensorDict)."""

  def __init__(self, policy: ActorCritic) -> None:
    super().__init__()
    self.obs_normalizer = policy.actor_obs_normalizer
    self.actor = policy.actor

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    x = self.obs_normalizer(obs)
    return self.actor(x)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  # scripts/ -> mujoco_playground/ -> repo root (mujoco_playground_g1_standing)
  repo_root = Path(__file__).resolve().parents[2]
  parser.add_argument(
      "--checkpoint",
      type=Path,
      default=repo_root / "model" / "model_7750.pt",
      help="Path to model_N.pt from RSL-RL.",
  )
  parser.add_argument(
      "--output",
      type=Path,
      default=repo_root / "model" / "g1_standing_policy.onnx",
      help="Output .onnx path.",
  )
  parser.add_argument(
      "--obs-groups",
      type=str,
      default='{"policy": ["state"], "critic": ["privileged_state"]}',
      help="JSON dict of obs groups (must match training).",
  )
  parser.add_argument(
      "--opset",
      type=int,
      default=18,
      help="ONNX opset (PyTorch 2.9+ often exports as 18 then may down-convert).",
  )
  parser.add_argument(
      "--no-verify",
      action="store_true",
      help="Skip onnxruntime check after export.",
  )
  args = parser.parse_args()

  ckpt_path = args.checkpoint.expanduser().resolve()
  if not ckpt_path.is_file():
    print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
    sys.exit(1)

  loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
  if "model_state_dict" not in loaded:
    print("Expected a dict with 'model_state_dict' (RSL-RL save format).", file=sys.stderr)
    sys.exit(1)
  sd = loaded["model_state_dict"]

  obs_groups = json.loads(args.obs_groups)
  policy = _build_actor_critic(sd, obs_groups, policy_extras=None)
  wrapper = ActorInferenceWrapper(policy)
  wrapper.eval()

  n_obs, n_act, n_crit = _infer_dims(sd)
  dummy = torch.zeros(1, n_obs, dtype=torch.float32)
  out_path = args.output.expanduser().resolve()
  out_path.parent.mkdir(parents=True, exist_ok=True)

  torch.onnx.export(
      wrapper,
      dummy,
      str(out_path),
      input_names=["obs"],
      output_names=["action"],
      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
      opset_version=args.opset,
      do_constant_folding=True,
  )

  meta = {
      "source_checkpoint": str(ckpt_path),
      "onnx_path": str(out_path),
      "obs_dim": n_obs,
      "action_dim": n_act,
      "critic_obs_dim": n_crit,
      "onnx_input_name": "obs",
      "onnx_output_name": "action",
      "obs_groups": obs_groups,
      "training_iteration": int(loaded.get("iter", -1)),
  }
  meta_path = out_path.with_suffix(".onnx.json")
  meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
  print(f"Wrote {out_path}")
  print(f"Wrote {meta_path}")
  print(json.dumps(meta, indent=2))

  if not args.no_verify:
    try:
      import numpy as np
      import onnxruntime as ort
    except ImportError as e:
      print(f"Skipping verify (install onnxruntime, numpy): {e}")
      return

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_in = {"obs": dummy.numpy().astype(np.float32)}
    ort_out = sess.run([meta["onnx_output_name"]], ort_in)[0]
    with torch.no_grad():
      torch_out = wrapper(dummy).numpy()
    err = float(np.max(np.abs(ort_out - torch_out)))
    print(f"onnxruntime vs torch max abs error: {err:.6e}")
    if err > 1e-4:
      print("Warning: error larger than 1e-4; check opset / dtypes.", file=sys.stderr)


if __name__ == "__main__":
  main()
