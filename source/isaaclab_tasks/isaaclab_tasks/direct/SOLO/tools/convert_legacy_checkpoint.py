"""Explicitly convert SKRL 1.x checkpoint module names; never run implicitly."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Convert an explicitly supplied SKRL 1.x AMP checkpoint")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=True)
    if "observation_preprocessor" in checkpoint:
        raise ValueError("Checkpoint already contains SKRL 2.x observation_preprocessor")
    if "state_preprocessor" not in checkpoint:
        raise ValueError("Legacy checkpoint has no state_preprocessor to convert")
    if "amp_state_preprocessor" in checkpoint:
        checkpoint["amp_observation_preprocessor"] = checkpoint.pop("amp_state_preprocessor")
    checkpoint["observation_preprocessor"] = checkpoint.pop("state_preprocessor")
    checkpoint["solo_legacy_conversion"] = {
        "source": str(Path(args.input).resolve()),
        "note": "module names converted explicitly; load compatibility still depends on model architecture",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"Converted checkpoint written to {args.output}")


if __name__ == "__main__":
    main()
