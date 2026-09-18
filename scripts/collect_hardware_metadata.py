#!/usr/bin/env python3
"""Capture an auditable serving-host hardware snapshot using stdlib only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.profiling_common import collect_runtime_metadata, write_profile  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("cuda", "ascend"), required=True)
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--physical-accelerators", type=int, required=True)
    parser.add_argument("--logical-accelerators", type=int, required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--notes")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.physical_accelerators, args.logical_accelerators) <= 0:
        raise SystemExit("accelerator counts must be positive")
    if args.logical_accelerators < args.physical_accelerators:
        raise SystemExit("logical accelerator count cannot be smaller than physical count")
    identity = {
        "platform": args.platform,
        "hardware_id": args.hardware_id,
        "physical_accelerator_count": args.physical_accelerators,
        "logical_accelerator_count": args.logical_accelerators,
        "logical_devices_per_physical_accelerator": (
            args.logical_accelerators / args.physical_accelerators),
        "notes": args.notes,
    }
    payload = {
        "schema_version": 1,
        "artifact_type": "serving_host_hardware_metadata",
        "hardware_identity": identity,
        "runtime_metadata": collect_runtime_metadata(
            model_path=args.model_path,
            benchmark={"hardware_identity": identity},
            project_root=PROJECT_ROOT,
        ),
    }
    write_profile(payload, args.output)
    print(f"saved hardware metadata to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
