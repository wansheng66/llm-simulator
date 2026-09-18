#!/usr/bin/env python3
"""Collect a Cartesian fixed-batch validation suite from vLLM."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def csv_ints(value: str) -> List[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector-mode", choices=("streaming", "offline"),
                        default="streaming",
                        help="HTTP server collector or strict offline enqueue barrier")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--stages", nargs="+", choices=("prefill", "decode"),
                        default=["prefill", "decode"])
    parser.add_argument("--batch-sizes", type=csv_ints, default=csv_ints("1,4,8"))
    parser.add_argument("--prefill-lengths", type=csv_ints,
                        default=csv_ints("128,512,1024"))
    parser.add_argument("--decode-kv-lengths", type=csv_ints,
                        default=csv_ints("128,512,1024"))
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-first-token-spread-ms", type=float, default=10.0)
    parser.add_argument("--max-scheduled-spread-ms", type=float, default=5.0,
                        help="offline scheduler-cohort limit")
    parser.add_argument("--submission-mode",
                        choices=("batched_prompt", "concurrent_requests"),
                        default="batched_prompt")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--deployment-config", type=Path, required=True)
    parser.add_argument("--hardware-metadata-file", type=Path)
    parser.add_argument("--server-command-file", type=Path)
    parser.add_argument("--resume", action="store_true",
                        help="Skip existing valid points whose shape and TP match")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def reusable_output(path: Path, args: argparse.Namespace, stage: str,
                    batch: int, length: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["configuration"]
    except (OSError, ValueError, KeyError, TypeError):
        return False
    collector_mode = getattr(args, "collector_mode", "streaming")
    expected_collector = (
        "vllm_fixed_batch_offline" if collector_mode == "offline"
        else "vllm_fixed_batch_streaming"
    )
    expected_submission = (
        "offline_enqueue_barrier" if collector_mode == "offline"
        else getattr(args, "submission_mode", "batched_prompt")
    )
    length_key = "prompt_length" if stage == "prefill" else "initial_kv_length"
    return (
        payload.get("schema_version") == 2 and
        payload.get("collector") == expected_collector and
        payload.get("valid") is True and
        payload.get("fixed_batch_valid") is True and
        payload.get("model") == args.model and
        config.get("stage") == stage and
        int(config.get("tp_size", -1)) == args.tp_size and
        int(config.get("batch_size", -1)) == batch and
        int(config.get(length_key, -1)) == length and
        config.get("submission_mode") == expected_submission and
        "request-unique first 16-token block" in config.get("prompt_policy", "")
    )


def write_manifest(args: argparse.Namespace, outputs: List[str]) -> Path:
    manifest = {
        "schema_version": 1,
        "experiment_type": "fixed_batch_suite_manifest",
        "collector_mode": args.collector_mode,
        "tp_size": args.tp_size,
        "submission_mode": (
            "offline_enqueue_barrier" if args.collector_mode == "offline"
            else args.submission_mode),
        "deployment_config": str(args.deployment_config.resolve()),
        "hardware_metadata_file": (
            str(args.hardware_metadata_file.resolve())
            if args.hardware_metadata_file else None),
        "files": outputs,
    }
    path = args.output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def run_offline_suite(args: argparse.Namespace) -> int:
    """Load one offline engine and collect every requested shape with it."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from scripts.collect_fixed_batch_vllm import exact_length_prompt
    from scripts.collect_fixed_batch_vllm_offline import collect_point, load_json

    deployment = load_json(args.deployment_config) or {}
    configured_tp = deployment.get(
        "tensor_parallel_size", deployment.get("tp_size"))
    if configured_tp is not None and int(configured_tp) != args.tp_size:
        raise SystemExit("deployment TP does not match --tp-size")
    if deployment.get("enable_prefix_caching") is True:
        raise SystemExit("strict baseline requires prefix caching disabled")
    max_num_seqs = int(deployment.get("max_num_seqs", max(args.batch_sizes)))
    max_num_batched_tokens = int(deployment.get(
        "max_num_batched_tokens", 8192))
    largest_batch = max(args.batch_sizes)
    largest_length = max(args.prefill_lengths + args.decode_kv_lengths)
    if (largest_batch > max_num_seqs
            or largest_batch * largest_length > max_num_batched_tokens):
        raise SystemExit(
            "at least one shape cannot fit one configured scheduler iteration")

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        tokenizer=tokenizer_path,
        tensor_parallel_size=args.tp_size,
        dtype=args.dtype,
        trust_remote_code=True,
        max_model_len=int(deployment.get("max_model_len", 8192)),
        gpu_memory_utilization=float(
            deployment.get("gpu_memory_utilization", 0.85)),
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        block_size=int(deployment.get(
            "block_size_tokens", deployment.get("block_size", 16))),
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        enforce_eager=bool(deployment.get("enforce_eager", True)),
        disable_log_stats=False,
        async_scheduling=False,
    )

    outputs = []
    stage_lengths = {
        "prefill": args.prefill_lengths,
        "decode": args.decode_kv_lengths,
    }
    for stage in args.stages:
        for batch in args.batch_sizes:
            for length in stage_lengths[stage]:
                output = args.output_dir / f"{stage}_b{batch}_l{length}.json"
                if args.resume and reusable_output(
                        output, args, stage, batch, length):
                    print(f"reuse valid {stage}: TP={args.tp_size}, "
                          f"batch={batch}, length={length}")
                    outputs.append(str(output.resolve()))
                    continue

                point_args = argparse.Namespace(**vars(args))
                point_args.stage = stage
                point_args.batch_size = batch
                point_args.length = length
                point_args.output = output
                base_prompt = exact_length_prompt(tokenizer, length)
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=(1 if stage == "prefill"
                                else args.decode_tokens),
                    ignore_eos=True,
                    detokenize=False,
                )
                print(f"collect {stage}: TP={args.tp_size}, "
                      f"batch={batch}, length={length}")
                payload = collect_point(
                    llm, sampling_params, base_prompt, point_args, deployment)
                if not payload["fixed_batch_valid"]:
                    raise SystemExit(
                        f"invalid fixed-batch point written to {output}")
                outputs.append(str(output.resolve()))

    path = write_manifest(args, outputs)
    print(f"saved {len(outputs)} fixed-batch points; manifest={path}")
    return 0


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.collector_mode == "offline":
        return run_offline_suite(args)
    outputs = []
    stage_lengths = {
        "prefill": args.prefill_lengths,
        "decode": args.decode_kv_lengths,
    }
    for stage in args.stages:
        lengths = stage_lengths[stage]
        for batch in args.batch_sizes:
            for length in lengths:
                output = args.output_dir / f"{stage}_b{batch}_l{length}.json"
                if args.resume and reusable_output(
                        output, args, stage, batch, length):
                    print(f"reuse valid {stage}: TP={args.tp_size}, "
                          f"batch={batch}, length={length}")
                    outputs.append(str(output.resolve()))
                    continue
                collector_script = (
                    "collect_fixed_batch_vllm_offline.py"
                    if args.collector_mode == "offline"
                    else "collect_fixed_batch_vllm.py"
                )
                command = [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / collector_script),
                    "--model", args.model,
                    "--tokenizer", args.tokenizer or args.model,
                    "--stage", stage,
                    "--tp-size", str(args.tp_size),
                    "--batch-size", str(batch),
                    "--length", str(length),
                    "--decode-tokens", str(args.decode_tokens),
                    "--warmup", str(args.warmup),
                    "--repeats", str(args.repeats),
                    "--max-first-token-spread-ms",
                    str(args.max_first_token_spread_ms),
                    "--dtype", args.dtype,
                    "--deployment-config", str(args.deployment_config),
                    "--output", str(output),
                ]
                if args.collector_mode == "offline":
                    command.extend([
                        "--max-scheduled-spread-ms",
                        str(args.max_scheduled_spread_ms),
                    ])
                else:
                    command.extend([
                        "--base-url", args.base_url,
                        "--api-key", args.api_key,
                        "--submission-mode", args.submission_mode,
                    ])
                if args.server_command_file and args.collector_mode == "streaming":
                    command.extend(["--server-command-file", str(args.server_command_file)])
                if args.hardware_metadata_file:
                    command.extend([
                        "--hardware-metadata-file", str(args.hardware_metadata_file)])
                print(f"collect {stage}: TP={args.tp_size}, batch={batch}, length={length}")
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)
                outputs.append(str(output.resolve()))
    path = write_manifest(args, outputs)
    print(f"saved {len(outputs)} fixed-batch points; manifest={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
