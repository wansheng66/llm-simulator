#!/usr/bin/env python3
"""Collect strict fixed-batch Prefill or Decode truth via vLLM offline LLM.

Unlike the OpenAI HTTP collector, this program enqueues the complete prompt
list before it lets the engine run.  A point is accepted only when vLLM returns
per-request engine timestamps and every sequence belongs to one narrow
scheduling/first-token cohort.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collect_fixed_batch_vllm import (  # noqa: E402
    exact_length_prompt,
    prompt_sha256,
    unique_prompt_ids,
)
from scripts.profiling_common import (  # noqa: E402
    SCHEMA_VERSION,
    collect_runtime_metadata,
    describe_ms,
    sha256_file,
    write_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="local model path visible inside this container")
    parser.add_argument("--tokenizer")
    parser.add_argument("--stage", choices=("prefill", "decode"), required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--length", type=int, required=True,
                        help="prompt length for Prefill; initial KV length for Decode")
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-scheduled-spread-ms", type=float, default=5.0)
    parser.add_argument("--max-first-token-spread-ms", type=float, default=10.0)
    parser.add_argument("--deployment-config", type=Path, required=True)
    parser.add_argument("--hardware-metadata-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0.0 else None


def metrics_snapshot(metrics: Any) -> Optional[Dict[str, Any]]:
    if metrics is None:
        return None
    if dataclasses.is_dataclass(metrics):
        values = dataclasses.asdict(metrics)
    elif hasattr(metrics, "__dict__"):
        values = vars(metrics)
    else:
        values = {
            name: getattr(metrics, name)
            for name in (
                "arrival_time", "queued_ts", "scheduled_ts", "first_token_ts",
                "last_token_ts", "first_token_latency", "num_generation_tokens",
            )
            if hasattr(metrics, name)
        }
    result = {}
    for key, value in values.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
    return result


def request_timing(output: Any) -> Dict[str, Any]:
    """Normalize vLLM 0.19 RequestStateStats into benchmark milliseconds."""
    metrics = getattr(output, "metrics", None)
    snapshot = metrics_snapshot(metrics)
    if metrics is None:
        return {"metrics": None, "error": "RequestOutput.metrics is absent"}

    scheduled = _number(getattr(metrics, "scheduled_ts", None))
    first = _number(getattr(metrics, "first_token_ts", None))
    last = _number(getattr(metrics, "last_token_ts", None))
    first_latency = _number(getattr(metrics, "first_token_latency", None))
    output_tokens = sum(
        len(getattr(completion, "token_ids", ()) or ())
        for completion in (getattr(output, "outputs", None) or [])
    )
    prefill_ms = None
    if scheduled is not None and first is not None and first >= scheduled:
        prefill_ms = (first - scheduled) * 1000.0
    elif first_latency is not None:
        # Compatibility fallback. This includes frontend queueing and is kept
        # visible in timing_source rather than silently treated as service time.
        prefill_ms = first_latency * 1000.0

    tpot_ms = None
    if (first is not None and last is not None and last >= first
            and output_tokens > 1):
        tpot_ms = (last - first) * 1000.0 / (output_tokens - 1)
    return {
        "metrics": snapshot,
        "scheduled_ts": scheduled,
        "first_token_ts": first,
        "last_token_ts": last,
        "prefill_ms": prefill_ms,
        "tpot_ms": tpot_ms,
        "actual_prompt_tokens": len(getattr(output, "prompt_token_ids", ()) or ()),
        "actual_output_tokens": output_tokens,
        "timing_source": (
            "scheduled_ts_to_first_token_ts"
            if scheduled is not None and first is not None else
            "first_token_latency_fallback"
        ),
        "error": None,
    }


def spread_ms(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return (max(values) - min(values)) * 1000.0


def run_batch(llm: Any, sampling_params: Any, base_prompt: List[int],
              args: argparse.Namespace, repeat_id: str) -> Dict[str, Any]:
    prompt_rows = []
    prompts = []
    for index in range(args.batch_size):
        request_id = f"offline_{args.stage}_{repeat_id}_{index + 1}"
        token_ids = unique_prompt_ids(base_prompt, request_id)
        prompts.append({"prompt_token_ids": token_ids})
        prompt_rows.append((request_id, prompt_sha256(token_ids)))

    # LLM.generate(list) is documented as automatic batching, but it still
    # combines request admission and engine execution. On vLLM-Ascend 0.19.1
    # this can let the engine schedule the first prompt before the rest of the
    # list has reached its queue. enqueue() is the explicit admission barrier:
    # it adds every request without starting processing, and only the following
    # wait_for_completion() call is allowed to run the engine.
    started = time.perf_counter()
    enqueued_request_ids = llm.enqueue(
        prompts, sampling_params, use_tqdm=False)
    admission_finished = time.perf_counter()
    outputs = llm.wait_for_completion(use_tqdm=False)
    wall_ms = (time.perf_counter() - started) * 1000.0
    enqueue_ms = (admission_finished - started) * 1000.0

    prompt_by_engine_id = {}
    for enqueued_request_id, prompt_row in zip(
            enqueued_request_ids, prompt_rows):
        enqueued_request_id = str(enqueued_request_id)
        prompt_by_engine_id[enqueued_request_id] = (
            prompt_row, enqueued_request_id)
        # vLLM-Ascend's in-process EngineCore may return an internal enqueue
        # ID such as "2-9459d2e9" while RequestOutput exposes the original
        # frontend ID "2". The numeric prefix is unique because LLM allocates
        # it from its monotonically increasing request counter.
        frontend_id = enqueued_request_id.split("-", 1)[0]
        if frontend_id.isdigit():
            prompt_by_engine_id[frontend_id] = (
                prompt_row, enqueued_request_id)
    requests = []
    for output in outputs:
        timing = request_timing(output)
        engine_request_id = str(getattr(output, "request_id", ""))
        prompt_match = prompt_by_engine_id.get(engine_request_id)
        if prompt_match is None:
            request_id = f"unknown_engine_request_{engine_request_id}"
            fingerprint = None
            enqueued_request_id = None
            timing["error"] = (
                "completed engine request ID was not returned by LLM.enqueue"
            )
        else:
            prompt_row, enqueued_request_id = prompt_match
            request_id, fingerprint = prompt_row
        timing.update({
            "request_id": request_id,
            "engine_request_id": engine_request_id,
            "enqueue_request_id": enqueued_request_id,
            "prompt_sha256": fingerprint,
            "status": "success" if timing["error"] is None else "failed",
        })
        requests.append(timing)

    scheduled_spread = spread_ms(requests, "scheduled_ts")
    first_token_spread = spread_ms(requests, "first_token_ts")
    if args.stage == "prefill":
        values = [row["prefill_ms"] for row in requests
                  if row.get("prefill_ms") is not None]
        observed = max(values) if len(values) == args.batch_size else None
        metric_name = "max_engine_prefill_service_ms"
    else:
        values = [row["tpot_ms"] for row in requests
                  if row.get("tpot_ms") is not None]
        observed = statistics.fmean(values) if len(values) == args.batch_size else None
        metric_name = "mean_engine_tpot_ms"
    return {
        "repeat_id": repeat_id,
        "successful_requests": sum(row["status"] == "success" for row in requests),
        "scheduled_spread_ms": scheduled_spread,
        "first_token_spread_ms": first_token_spread,
        "enqueue_time_ms": enqueue_ms,
        "enqueued_request_ids": [str(item) for item in enqueued_request_ids],
        "batch_wall_time_ms": wall_ms,
        "metric_name": metric_name,
        "observed_iteration_ms": observed,
        "requests": requests,
    }


def load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_point(llm: Any, sampling_params: Any, base_prompt: List[int],
                  args: argparse.Namespace,
                  deployment: Dict[str, Any]) -> Dict[str, Any]:
    """Collect, validate, write and return one shape using an existing LLM."""
    output_tokens = 1 if args.stage == "prefill" else args.decode_tokens
    warmups = [run_batch(llm, sampling_params, base_prompt, args,
                         f"warmup_{index + 1}")
               for index in range(args.warmup)]
    samples = []
    for index in range(args.repeats):
        sample = run_batch(llm, sampling_params, base_prompt, args,
                           f"repeat_{index + 1}")
        samples.append(sample)
        print(f"repeat {index + 1}/{args.repeats}: "
              f"{sample['observed_iteration_ms']} ms; "
              f"scheduled_spread={sample['scheduled_spread_ms']} ms")

    values = [float(row["observed_iteration_ms"]) for row in samples
              if row.get("observed_iteration_ms") is not None]
    scheduled_spreads = [float(row["scheduled_spread_ms"]) for row in samples
                         if row.get("scheduled_spread_ms") is not None]
    first_spreads = [float(row["first_token_spread_ms"]) for row in samples
                     if row.get("first_token_spread_ms") is not None]
    timing_complete = len(values) == args.repeats
    scheduled_cohort_valid = (
        len(scheduled_spreads) == args.repeats
        and max(scheduled_spreads) <= args.max_scheduled_spread_ms
    )
    first_token_cohort_valid = (
        len(first_spreads) == args.repeats
        and max(first_spreads) <= args.max_first_token_spread_ms
    )
    output_shape_errors = []
    for row in samples:
        if row["successful_requests"] != args.batch_size:
            output_shape_errors.append({
                "repeat_id": row["repeat_id"],
                "error": "successful request count does not match batch size",
                "expected": args.batch_size,
                "actual": row["successful_requests"],
            })
        for request in row["requests"]:
            if (request["actual_prompt_tokens"] != args.length
                    or request["actual_output_tokens"] != output_tokens):
                output_shape_errors.append({
                    "repeat_id": row["repeat_id"],
                    "request_id": request["request_id"],
                    "engine_request_id": request["engine_request_id"],
                    "status": request["status"],
                    "error": request["error"],
                    "expected_prompt_tokens": args.length,
                    "actual_prompt_tokens": request["actual_prompt_tokens"],
                    "expected_output_tokens": output_tokens,
                    "actual_output_tokens": request["actual_output_tokens"],
                })
    output_valid = not output_shape_errors
    fixed_batch_valid = (
        timing_complete and output_valid and scheduled_cohort_valid
        and first_token_cohort_valid
    )
    benchmark = {
        "kind": "vllm_offline_fixed_batch_iteration_validation",
        "stage": args.stage,
        "tp_size": args.tp_size,
        "batch_size": args.batch_size,
        "prompt_length" if args.stage == "prefill" else "initial_kv_length": args.length,
        "decode_tokens": output_tokens,
        "dtype": args.dtype,
        "warmup_batches": args.warmup,
        "measured_batches": args.repeats,
        "submission_mode": "offline_enqueue_barrier",
        "release_policy": (
            "complete prompt list added by LLM.enqueue before "
            "LLM.wait_for_completion starts engine processing"),
        "v1_engine_core_multiprocessing": False,
        "async_scheduling": False,
        "enable_chunked_prefill": False,
        "prompt_policy": (
            "exact-length token-id prompts with a request-unique first 16-token "
            "block to prevent prefix-cache reuse"),
        "metric": (
            "maximum scheduled-to-first-token engine time"
            if args.stage == "prefill" else
            "mean per-request engine TPOT"),
        "max_allowed_scheduled_spread_ms": args.max_scheduled_spread_ms,
        "max_allowed_first_token_spread_ms": args.max_first_token_spread_ms,
    }
    hardware = load_json(args.hardware_metadata_file)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "collector": "vllm_fixed_batch_offline",
        "experiment_type": "fixed_batch_iteration_validation",
        "model": args.model,
        "configuration": benchmark,
        "deployment": {
            "config_path": str(args.deployment_config.resolve()),
            "config_sha256": sha256_file(args.deployment_config),
            "config": deployment,
        },
        "metadata": collect_runtime_metadata(
            model_path=args.model, benchmark=benchmark, project_root=PROJECT_ROOT),
        "serving_host_hardware": ({
            "path": str(args.hardware_metadata_file.resolve()),
            "sha256": sha256_file(args.hardware_metadata_file),
            "snapshot": hardware,
        } if args.hardware_metadata_file else None),
        "warmup": warmups,
        "samples": samples,
        "summary": describe_ms(values) if values else None,
        "valid": fixed_batch_valid,
        "fixed_batch_valid": fixed_batch_valid,
        "validity": {
            "timing_metrics_complete": timing_complete,
            "output_shape_valid": output_valid,
            "output_shape_errors": output_shape_errors,
            "scheduled_cohort_valid": scheduled_cohort_valid,
            "first_token_cohort_valid": first_token_cohort_valid,
            "max_observed_scheduled_spread_ms": (
                max(scheduled_spreads) if scheduled_spreads else None),
            "max_observed_first_token_spread_ms": (
                max(first_spreads) if first_spreads else None),
        },
        "limitations": [
            "Offline engine-service truth excludes HTTP, tokenization and client overhead.",
            "Decode TPOT spans a growing KV length; --length is the initial KV length.",
            "Cross-hardware comparisons must use this same offline protocol on every device.",
        ],
    }
    write_profile(payload, args.output)
    print(f"saved offline fixed-batch run to {args.output}; "
          f"fixed_batch_valid={fixed_batch_valid}")
    return payload


def main() -> int:
    args = parse_args()
    if min(args.tp_size, args.batch_size, args.length, args.repeats) <= 0:
        raise SystemExit("TP, batch, length and repeats must be positive")
    if args.stage == "decode" and args.decode_tokens < 2:
        raise SystemExit("Decode validation needs at least two output tokens")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # enqueue() is an admission barrier only inside one LLM engine process.
    # With the default V1 EngineCore subprocess, the core may consume and
    # schedule the first IPC message while the frontend is still sending the
    # rest of the logical batch. Keep the scheduler in this process so every
    # enqueue call completes before wait_for_completion starts the engine.
    # This does not disable TP worker multiprocessing.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    deployment = load_json(args.deployment_config) or {}
    configured_tp = deployment.get("tensor_parallel_size", deployment.get("tp_size"))
    if configured_tp is not None and int(configured_tp) != args.tp_size:
        raise SystemExit("deployment TP does not match --tp-size")
    max_num_seqs = int(deployment.get("max_num_seqs", args.batch_size))
    max_num_batched_tokens = int(deployment.get(
        "max_num_batched_tokens", max(args.batch_size * args.length, 8192)))
    query_tokens = args.batch_size * args.length if args.stage == "prefill" else args.batch_size
    if args.batch_size > max_num_seqs or query_tokens > max_num_batched_tokens:
        raise SystemExit("shape cannot fit one configured scheduler iteration")
    if deployment.get("enable_prefix_caching") is True:
        raise SystemExit("strict baseline requires prefix caching disabled")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    base_prompt = exact_length_prompt(tokenizer, args.length)
    llm = LLM(
        model=args.model,
        tokenizer=tokenizer_path,
        tensor_parallel_size=args.tp_size,
        dtype=args.dtype,
        trust_remote_code=True,
        max_model_len=int(deployment.get("max_model_len", 8192)),
        gpu_memory_utilization=float(deployment.get("gpu_memory_utilization", 0.85)),
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        block_size=int(deployment.get("block_size_tokens",
                                      deployment.get("block_size", 16))),
        enable_prefix_caching=False,
        # Every accepted shape is already required to fit completely inside
        # max_num_batched_tokens. Disabling chunked prefill prevents the default
        # max_num_partial_prefills=1 policy from serializing the logical batch.
        enable_chunked_prefill=False,
        enforce_eager=bool(deployment.get("enforce_eager", True)),
        disable_log_stats=False,
        # Strict fixed-batch truth requires all prompts to be enqueued before
        # the engine begins a scheduler step. vLLM-Ascend otherwise enables
        # asynchronous scheduling by default and may admit the prompt list in
        # separate iterations.
        async_scheduling=False,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=(1 if args.stage == "prefill" else args.decode_tokens),
        ignore_eos=True,
        detokenize=False,
    )
    payload = collect_point(
        llm, sampling_params, base_prompt, args, deployment)
    return 0 if payload["fixed_batch_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
