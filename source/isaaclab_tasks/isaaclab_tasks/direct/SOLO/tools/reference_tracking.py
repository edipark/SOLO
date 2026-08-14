"""Measure joint/body consistency while replaying a G1 reference motion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from ..schema import G1_JOINT_NAMES, G1_KEY_BODY_NAMES
except ImportError:
    from isaaclab_tasks.direct.SOLO.schema import G1_JOINT_NAMES, G1_KEY_BODY_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True)
    parser.add_argument("--tracked-motion", default=None, help="Optional recorded NPZ to compare against the reference")
    parser.add_argument("--output", default="reference_tracking.json")
    args = parser.parse_args()
    reference = np.load(args.motion)
    if tuple(reference["dof_names"].tolist()) != G1_JOINT_NAMES:
        raise ValueError("Reference does not use the SOLO G1 joint schema")
    result = {
        "motion": str(Path(args.motion).resolve()),
        "frames": int(reference["dof_positions"].shape[0]),
        "fps": int(reference["fps"]),
        "joint_names_valid": True,
        "key_bodies_valid": all(name in reference["body_names"] for name in G1_KEY_BODY_NAMES),
    }
    if args.tracked_motion:
        tracked = np.load(args.tracked_motion)
        count = min(len(reference["dof_positions"]), len(tracked["dof_positions"]))
        reference_ids = [reference["dof_names"].tolist().index(name) for name in G1_JOINT_NAMES]
        tracked_ids = [tracked["dof_names"].tolist().index(name) for name in G1_JOINT_NAMES]
        error = tracked["dof_positions"][:count, tracked_ids] - reference["dof_positions"][:count, reference_ids]
        result.update(
            joint_mae=float(np.abs(error).mean()), joint_rmse=float(np.square(error).mean() ** 0.5),
            per_joint_rmse={
                name: float(value)
                for name, value in zip(G1_JOINT_NAMES, np.square(error).mean(axis=0) ** 0.5)
            },
        )
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
