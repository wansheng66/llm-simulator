#!/usr/bin/env python3
"""Collect every BenchmarkSpec case from one running vLLM deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.benchmark_protocol import load_benchmark_spec  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reusable(path: Path, model: str, tp: int, stage: str,
             batch: int, length: int, repeats: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["configuration"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    length_key = "prompt_length" if stage == "prefill" else "initial_kv_length"
    return (
        payload.get("collector") == "vllm_fixed_batch_streaming"
        and payload.get("valid") is True
        and payload.get("fixed_batch_valid") is True
        and payload.get("model") == model
        and config.get("stage") == stage
        and int(config.get("tp_size", -1)) == tp
        and int(config.get("batch_size", -1)) == batch
        and int(config.get(length_key, -1)) == length
        and int(config.get("measured_batches", -1)) == repeats
        and config.get("submission_mode") == "batched_prompt"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True,
                        help="stable logical model id; paths may differ between machines")
    parser.add_argument("--tokenizer")
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--benchmark-spec", type=Path, required=True)
    parser.add_argument("--deployment-config", type=Path, required=True)
    parser.add_argument("--hardware-metadata-file", type=Path)
    parser.add_argument("--server-command-file", type=Path)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-start-skew-ms", type=float, default=75.0)
    parser.add_argument("--max-first-token-spread-ms", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = load_benchmark_spec(args.benchmark_spec)
    if spec["model"]["model_id"] != args.model_id:
        raise SystemExit("--model-id does not match BenchmarkSpec model_id")
    if spec["model"]["dtype"] != args.dtype:
        raise SystemExit("--dtype does not match BenchmarkSpec dtype")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for stage in ("prefill", "decode"):
        length_key = "prompt_length" if stage == "prefill" else "kv_length"
        for case in spec["workloads"][stage]:
            name = case["case"]
            batch = int(case["batch_size"])
            length = int(case[length_key])
            output = args.output_dir / f"{stage}_{name}.json"
            if not (args.resume and reusable(
                    output, args.model, args.tp_size, stage,
                    batch, length, args.repeats)):
                command = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "collect_fixed_batch_vllm.py"),
                    "--base-url", args.base_url,
                    "--api-key", args.api_key,
                    "--model", args.model,
                    "--tokenizer", args.tokenizer or args.model,
                    "--stage", stage,
                    "--tp-size", str(args.tp_size),
                    "--batch-size", str(batch),
                    "--length", str(length),
                    "--decode-tokens", str(args.decode_tokens),
                    "--warmup", str(args.warmup),
                    "--repeats", str(args.repeats),
                    "--submission-mode", "batched_prompt",
                    "--max-start-skew-ms", str(args.max_start_skew_ms),
                    "--max-first-token-spread-ms",
                    str(args.max_first_token_spread_ms),
                    "--dtype", args.dtype,
                    "--deployment-config", str(args.deployment_config),
                    "--output", str(output),
                ]
                if args.server_command_file:
                    command.extend(["--server-command-file", str(args.server_command_file)])
                if args.hardware_metadata_file:
                    command.extend([
                        "--hardware-metadata-file", str(args.hardware_metadata_file)])
                print(f"collect {args.hardware_id} {name}: {stage} B={batch} L={length}")
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
            else:
                print(f"reuse {args.hardware_id} {name}")
            entries.append({
                "stage": stage,
                "case": name,
                "batch_size": batch,
                length_key: length,
                "file": str(output.resolve()),
            })

    manifest = {
        "schema_version": 1,
        "experiment_type": "relative_ground_truth_run",
        "protocol_version": spec["protocol_version"],
        "hardware_id": args.hardware_id,
        "model": args.model,
        "model_id": args.model_id,
        "tp_size": args.tp_size,
        "dtype": args.dtype,
        "benchmark_spec": str(args.benchmark_spec.resolve()),
        "benchmark_spec_sha256": sha256(args.benchmark_spec),
        "deployment_config": str(args.deployment_config.resolve()),
        "deployment_config_sha256": sha256(args.deployment_config),
        "hardware_metadata_file": (
            str(args.hardware_metadata_file.resolve())
            if args.hardware_metadata_file else None),
        "hardware_metadata_sha256": (
            sha256(args.hardware_metadata_file)
            if args.hardware_metadata_file else None),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "submission_mode": "batched_prompt",
        "entries": entries,
    }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(f"saved {len(entries)} cases; manifest={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
