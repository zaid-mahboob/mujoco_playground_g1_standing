#!/usr/bin/env python3
"""Load G1, apply standing leg pose, print support ratio, and open viewer."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import base as g1_base


# Must match `g1_standing.py` default_config().leg_pose
LEG_POSE = np.array(
    [
        -0.2, 0.0, 0.0, 0.59, -0.34, 0.0,  # left leg
        -0.2, 0.0, 0.0, 0.59, -0.34, 0.0,  # right leg
    ],
    dtype=float,
)


def _scene_xml_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return (
        repo_root
        / "mujoco_playground"
        / "_src"
        / "locomotion"
        / "g1"
        / "xmls"
        / "scene_mjx_feetonly_flat_terrain.xml"
    )


def _compute_support_ratio(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    torso_body_id = model.body("torso_link").id
    left_foot_site_id = model.site("left_foot").id
    right_foot_site_id = model.site("right_foot").id

    com_xyz = data.subtree_com[torso_body_id].copy()
    left_xyz = data.site_xpos[left_foot_site_id].copy()
    right_xyz = data.site_xpos[right_foot_site_id].copy()

    com_xy = com_xyz[:2]
    left_xy = left_xyz[:2]
    right_xy = right_xyz[:2]

    d_left = float(np.linalg.norm(com_xy - left_xy))
    d_right = float(np.linalg.norm(com_xy - right_xy))
    denom = d_left + d_right
    if denom < 1e-9:
        left_ratio = right_ratio = 0.5
    else:
        # Static two-support distribution along the COM line:
        # left foot takes more load when COM is closer to left foot.
        left_ratio = d_right / denom
        right_ratio = d_left / denom

    return com_xyz, left_xyz, right_xyz, left_ratio, right_ratio


def main() -> None:
    xml_path = _scene_xml_path()
    if not xml_path.exists():
        raise FileNotFoundError(f"Could not find model XML: {xml_path}")

    mjx_env.ensure_menagerie_exists()
    assets = g1_base.get_assets()
    model = mujoco.MjModel.from_xml_string(xml_path.read_text(), assets=assets)
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
    if key_id < 0:
        raise ValueError("Keyframe 'knees_bent' not found in XML.")

    mujoco.mj_resetDataKeyframe(model, data, key_id)

    # Upright pelvis (identity quaternion), legs at standing pose, upper body zero.
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qpos[7:] = 0.0
    data.qpos[7 : 7 + 12] = LEG_POSE

    # Refresh kinematics/COM with this static qpos.
    mujoco.mj_forward(model, data)

    com_xyz, left_xyz, right_xyz, left_ratio, right_ratio = _compute_support_ratio(
        model, data
    )

    print("\n=== G1 Standing Pose (single leg_pose + all other joints zero) ===")
    print(f"Model XML: {xml_path}")
    print(f"COM xyz (m):          {com_xyz}")
    print(f"Left foot xyz (m):    {left_xyz}")
    print(f"Right foot xyz (m):   {right_xyz}")
    print(f"Left support ratio:   {left_ratio:.4f} ({left_ratio*100:.2f}%)")
    print(f"Right support ratio:  {right_ratio:.4f} ({right_ratio*100:.2f}%)")

    print(
        "\nInterpretation: under quasi-static gravity compensation, "
        "the feet would contribute approximately in the above ratio."
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\nViewer opened. Close the viewer window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
