"""Regenerate all ablation artifacts from a raw JSONL file."""

import argparse
import json

try:
    from .reporting import generate_report
except ImportError:
    from reporting import generate_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_jsonl")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    print(json.dumps(generate_report(args.raw_jsonl, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
