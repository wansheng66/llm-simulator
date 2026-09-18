#!/usr/bin/env python3
"""Collect fixed-batch vLLM truth for one Prefill or Decode shape.

All requests in a repeat are released together. Prefill uses time-to-first-token
with one generated token; Decode uses per-request TPOT over several generated
tokens. This is an end-to-end iteration validation artifact, not an operator
profile and not an online-arrival workload.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.profiling_common import (  # noqa: E402
    SCHEMA_VERSION,
    collect_runtime_metadata,
    describe_ms,
    sha256_file,
    write_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--stage", choices=("prefill", "decode"), required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--length", type=int, required=True,
                        help="Prompt length for Prefill; initial KV length for Decode")
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16",
                        help="Server model dtype recorded for reproducibility")
    parser.add_argument("--warmup", type=int, default=1, help="Full-batch warmup repetitions")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-start-skew-ms", type=float, default=50.0)
    parser.add_argument(
        "--max-first-token-spread-ms", type=float, default=10.0,
        help=(
            "Maximum allowed spread between the first output token of each "
            "sequence. A larger spread indicates that joint HTTP submission "
            "did not become one fixed engine batch."
        ),
    )
    parser.add_argument(
        "--submission-mode",
        choices=("batched_prompt", "concurrent_requests"),
        default="batched_prompt",
        help=(
            "batched_prompt sends every sequence in one OpenAI request and is "
            "required for strict fixed-batch ground truth; concurrent_requests "
            "is retained only for compatibility with legacy measurements"
        ),
    )
    parser.add_argument("--deployment-config", type=Path,
                        help="JSON containing the exact vLLM launch/runtime configuration")
    parser.add_argument(
        "--hardware-metadata-file", type=Path,
        help="host-side hardware snapshot produced by collect_hardware_metadata.py",
    )
    parser.add_argument("--server-command-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def exact_length_prompt(tokenizer, target: int) -> List[int]:
    seed = tokenizer.encode(
        "The quick brown fox describes an inference system. ",
        add_special_tokens=False)
    if not seed or target <= 0:
        raise ValueError("tokenizer seed and target prompt length must be positive")
    return (seed * (target // len(seed) + 1))[:target]


def unique_prompt_ids(base: List[int], namespace: str) -> List[int]:
    """Give every request a distinct first token block without changing length."""
    if not base:
        raise ValueError("cannot uniquify an empty prompt")
    result = list(base)
    pool = sorted(set(base))
    if len(pool) < 2:
        raise ValueError("prompt seed needs at least two distinct token ids")
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    for index in range(min(16, len(result))):
        result[index] = pool[digest[index] % len(pool)]
    return result


def prompt_sha256(prompt_ids: List[int]) -> str:
    encoded = ",".join(str(item) for item in prompt_ids).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


async def one_request(client, request, prompt_length: int, output_tokens: int,
                      prompt_fingerprint: str, request_id: str,
                      release: asyncio.Event) -> Dict:
    """Send an already-built HTTP request after the batch release barrier.

    Building an OpenAI request serializes the (potentially long) token-id
    prompt.  Doing that after releasing the barrier creates artificial skew
    proportional to batch size.  The caller therefore prepares every request
    before scheduling this coroutine.
    """
    await release.wait()
    started = time.perf_counter()
    first_token = None
    last_token = None
    usage = None
    error = None
    try:
        response = await client.send(request, stream=True)
        if response.is_error:
            await response.aread()
            response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            chunk = json.loads(raw)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if choices and choices[0].get("text"):
                now = time.perf_counter()
                first_token = first_token or now
                last_token = now
        await response.aclose()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    completion_tokens = int((usage or {}).get("completion_tokens", 0) or 0)
    ttft_ms = (first_token - started) * 1000 if first_token else None
    tpot_ms = ((last_token - first_token) * 1000 / (completion_tokens - 1)
               if first_token and last_token and completion_tokens > 1 else None)
    return {
        "request_id": request_id,
        "prompt_sha256": prompt_fingerprint,
        "status": "success" if error is None and first_token else "failed",
        "error": error or (None if first_token else "response contained no output token"),
        "request_start_monotonic_s": started,
        "actual_prompt_tokens": int((usage or {}).get("prompt_tokens", 0) or
                                    prompt_length),
        "actual_output_tokens": completion_tokens,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": (ended - started) * 1000,
    }


def _choice_index(choice: Dict, batch_size: int) -> Optional[int]:
    try:
        index = int(choice.get("index"))
    except (TypeError, ValueError):
        return None
    return index if 0 <= index < batch_size else None


async def run_batched_prompt(client, args: argparse.Namespace,
                             prompt_ids: List[int], repeat_id: str) -> Dict:
    """Submit the complete logical batch in one OpenAI Completions request."""
    output_tokens = 1 if args.stage == "prefill" else args.decode_tokens
    batch_request_id = (
        f"fixed_{args.stage}_b{args.batch_size}_l{len(prompt_ids)}_{repeat_id}")
    prompts = []
    request_ids = []
    for index in range(args.batch_size):
        request_id = f"{batch_request_id}_{index + 1}"
        prompt = unique_prompt_ids(prompt_ids, request_id)
        prompts.append(prompt)
        request_ids.append((request_id, prompt_sha256(prompt)))

    request = client.build_request("POST", "completions", json={
        "model": args.model,
        "prompt": prompts,
        "max_tokens": output_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "request_id": batch_request_id,
        "ignore_eos": True,
    })
    started = time.perf_counter()
    first_tokens: List[Optional[float]] = [None] * args.batch_size
    last_tokens: List[Optional[float]] = [None] * args.batch_size
    usage = None
    error = None
    try:
        response = await client.send(request, stream=True)
        if response.is_error:
            await response.aread()
            response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if not raw or raw == "[DONE]":
                continue
            chunk = json.loads(raw)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                index = _choice_index(choice, args.batch_size)
                if index is None or not choice.get("text"):
                    continue
                now = time.perf_counter()
                if first_tokens[index] is None:
                    first_tokens[index] = now
                last_tokens[index] = now
        await response.aclose()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()

    requests = []
    for index, (request_id, fingerprint) in enumerate(request_ids):
        first_token = first_tokens[index]
        last_token = last_tokens[index]
        request_error = error or (
            None if first_token is not None else
            "batched response contained no output token for this choice")
        requests.append({
            "request_id": request_id,
            "choice_index": index,
            "prompt_sha256": fingerprint,
            "status": "success" if request_error is None else "failed",
            "error": request_error,
            "request_start_monotonic_s": started,
            "actual_prompt_tokens": len(prompt_ids),
            "actual_output_tokens": output_tokens if request_error is None else 0,
            "ttft_ms": ((first_token - started) * 1000
                        if first_token is not None else None),
            "tpot_ms": (
                (last_token - first_token) * 1000 / (output_tokens - 1)
                if first_token is not None and last_token is not None
                and output_tokens > 1 else None
            ),
            "e2e_ms": (ended - started) * 1000,
        })

    successful = [row for row in requests if row["status"] == "success"]
    if args.stage == "prefill":
        metrics = [float(row["ttft_ms"]) for row in successful
                   if row["ttft_ms"] is not None]
        observed = max(metrics) if len(metrics) == args.batch_size else None
        metric_name = "batch_time_to_all_first_tokens_ms"
    else:
        metrics = [float(row["tpot_ms"]) for row in successful
                   if row["tpot_ms"] is not None]
        observed = (statistics.fmean(metrics)
                    if len(metrics) == args.batch_size else None)
        metric_name = "mean_request_tpot_ms"
    first_values = [value for value in first_tokens if value is not None]
    return {
        "repeat_id": repeat_id,
        "batch_request_id": batch_request_id,
        "release_monotonic_s": started,
        "request_start_skew_ms": 0.0,
        "first_token_spread_ms": (
            (max(first_values) - min(first_values)) * 1000
            if first_values else None
        ),
        "successful_requests": len(successful),
        "metric_name": metric_name,
        "observed_iteration_ms": observed,
        "server_usage": usage,
        "requests": requests,
    }


async def run_concurrent_requests(client, args: argparse.Namespace,
                                  prompt_ids: List[int], repeat_id: str) -> Dict:
    output_tokens = 1 if args.stage == "prefill" else args.decode_tokens
    release = asyncio.Event()
    prepared = []
    request_ids = []
    for index in range(args.batch_size):
        request_id = (
            f"fixed_{args.stage}_b{args.batch_size}_l{len(prompt_ids)}_"
            f"{repeat_id}_{index + 1}")
        request_prompt_ids = unique_prompt_ids(prompt_ids, request_id)
        request = client.build_request("POST", "completions", json={
            "model": args.model,
            "prompt": request_prompt_ids,
            "max_tokens": output_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "request_id": request_id,
            "ignore_eos": True,
        })
        prepared.append(request)
        request_ids.append((request_id, prompt_sha256(request_prompt_ids)))
    tasks = [asyncio.create_task(one_request(
        client, request, len(prompt_ids), output_tokens, fingerprint,
        request_id, release))
        for request, (request_id, fingerprint) in zip(prepared, request_ids)]
    await asyncio.sleep(0)
    released = time.perf_counter()
    release.set()
    requests = await asyncio.gather(*tasks)
    successful = [row for row in requests if row["status"] == "success"]
    starts = [row["request_start_monotonic_s"] for row in requests]
    if args.stage == "prefill":
        metrics = [float(row["ttft_ms"]) for row in successful
                   if row["ttft_ms"] is not None]
        observed = max(metrics) if len(metrics) == args.batch_size else None
        metric_name = "batch_time_to_all_first_tokens_ms"
    else:
        metrics = [float(row["tpot_ms"]) for row in successful
                   if row["tpot_ms"] is not None]
        observed = statistics.fmean(metrics) if len(metrics) == args.batch_size else None
        metric_name = "mean_request_tpot_ms"
    return {
        "repeat_id": repeat_id,
        "release_monotonic_s": released,
        "request_start_skew_ms": (max(starts) - min(starts)) * 1000 if starts else 0.0,
        "successful_requests": len(successful),
        "metric_name": metric_name,
        "observed_iteration_ms": observed,
        "requests": requests,
    }


async def run_batch(client, args: argparse.Namespace, prompt_ids: List[int],
                    repeat_id: str) -> Dict:
    if args.submission_mode == "batched_prompt":
        return await run_batched_prompt(client, args, prompt_ids, repeat_id)
    return await run_concurrent_requests(client, args, prompt_ids, repeat_id)


def load_deployment(args: argparse.Namespace) -> Dict:
    result = {}
    if args.deployment_config:
        config = json.loads(args.deployment_config.read_text(encoding="utf-8"))
        configured_tp = config.get("tensor_parallel_size", config.get("tp_size"))
        if configured_tp is not None and int(configured_tp) != args.tp_size:
            raise ValueError("deployment TP does not match --tp-size")
        configured_model = config.get("model")
        if configured_model is not None and str(configured_model) != args.model:
            raise ValueError("deployment model does not match --model")
        if config.get("enable_prefix_caching") is True:
            raise ValueError("fixed-batch baseline requires prefix caching to be disabled")
        max_sequences = config.get("max_num_seqs")
        if max_sequences is not None and int(max_sequences) < args.batch_size:
            raise ValueError("deployment max_num_seqs is smaller than --batch-size")
        result["config_path"] = str(args.deployment_config.resolve())
        result["config_sha256"] = sha256_file(args.deployment_config)
        result["config"] = config
    if args.server_command_file:
        result["server_command_file"] = str(args.server_command_file.resolve())
        result["server_command"] = args.server_command_file.read_text(
            encoding="utf-8", errors="replace").strip()
    return result


def load_hardware_metadata(args: argparse.Namespace) -> Optional[Dict]:
    if not args.hardware_metadata_file:
        return None
    payload = json.loads(args.hardware_metadata_file.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "serving_host_hardware_metadata":
        raise ValueError("unsupported --hardware-metadata-file artifact_type")
    return {
        "path": str(args.hardware_metadata_file.resolve()),
        "sha256": sha256_file(args.hardware_metadata_file),
        "snapshot": payload,
    }


def fixed_batch_eligibility(args: argparse.Namespace, deployment: Dict) -> Dict:
    """State whether the requested shape can be one scheduler iteration."""
    reasons = []
    config = deployment.get("config") or {}
    if args.submission_mode != "batched_prompt":
        reasons.append("sequences were submitted as independent HTTP requests")
    max_sequences = config.get("max_num_seqs")
    if max_sequences is None:
        reasons.append("deployment max_num_seqs is not recorded")
    elif int(max_sequences) < args.batch_size:
        reasons.append("batch_size exceeds deployment max_num_seqs")
    max_tokens = config.get("max_num_batched_tokens")
    query_tokens = (args.batch_size * args.length
                    if args.stage == "prefill" else args.batch_size)
    if max_tokens is None:
        reasons.append("deployment max_num_batched_tokens is not recorded")
    elif query_tokens > int(max_tokens):
        reasons.append(
            f"query token demand {query_tokens} exceeds max_num_batched_tokens "
            f"{int(max_tokens)}")
    return {
        "eligible": not reasons,
        "submission_mode": args.submission_mode,
        "query_tokens": query_tokens,
        "max_num_batched_tokens": max_tokens,
        "max_num_seqs": max_sequences,
        "reasons": reasons,
    }


def sample_usage_matches(sample: Dict, args: argparse.Namespace) -> bool:
    if args.submission_mode != "batched_prompt":
        return False
    usage = sample.get("server_usage") or {}
    expected_output = args.batch_size * (
        1 if args.stage == "prefill" else args.decode_tokens)
    return (
        int(usage.get("prompt_tokens", -1)) == args.batch_size * args.length
        and int(usage.get("completion_tokens", -1)) == expected_output
    )


async def main_async(args: argparse.Namespace) -> int:
    if min(args.tp_size, args.batch_size, args.length, args.repeats) <= 0:
        raise SystemExit("TP, batch, length and repeats must be positive")
    if args.stage == "decode" and args.decode_tokens < 2:
        raise SystemExit("Decode validation needs at least two output tokens")
    deployment = load_deployment(args)
    eligibility = fixed_batch_eligibility(args, deployment)
    if args.submission_mode == "batched_prompt" and not eligibility["eligible"]:
        raise SystemExit(
            "shape is not eligible for strict fixed-batch collection: "
            + "; ".join(eligibility["reasons"]))

    import httpx
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, trust_remote_code=True)
    prompt_ids = exact_length_prompt(tokenizer, args.length)
    base_url = args.base_url.rstrip("/") + "/"
    async with httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {args.api_key}"},
            timeout=args.timeout,
            trust_env=False) as client:
        response = await client.get("models")
        response.raise_for_status()
        warmups = []
        for index in range(args.warmup):
            warmups.append(await run_batch(
                client, args, prompt_ids, f"warmup_{index + 1}"))
        samples = []
        for index in range(args.repeats):
            sample = await run_batch(client, args, prompt_ids, f"repeat_{index + 1}")
            samples.append(sample)
            print(f"repeat {index + 1}/{args.repeats}: "
                  f"{sample['observed_iteration_ms']} ms")

    values = [float(row["observed_iteration_ms"]) for row in samples
              if row["observed_iteration_ms"] is not None]
    max_start_skew = max((row["request_start_skew_ms"] for row in samples), default=0.0)
    first_token_spreads = [
        float(row["first_token_spread_ms"])
        for row in samples if row.get("first_token_spread_ms") is not None
    ]
    max_first_token_spread = (
        max(first_token_spreads) if first_token_spreads else None
    )
    transport_valid = (
        len(values) == args.repeats
        and max_start_skew <= args.max_start_skew_ms
        and all(row["successful_requests"] == args.batch_size for row in samples)
    )
    usage_valid = all(sample_usage_matches(row, args) for row in samples)
    first_token_cohort_valid = (
        len(first_token_spreads) == args.repeats
        and max_first_token_spread is not None
        and max_first_token_spread <= args.max_first_token_spread_ms
    )
    fixed_batch_valid = (
        transport_valid and eligibility["eligible"] and usage_valid
        and first_token_cohort_valid
        and args.submission_mode == "batched_prompt"
    )
    benchmark = {
        "kind": "vllm_fixed_batch_iteration_validation",
        "stage": args.stage,
        "tp_size": args.tp_size,
        "batch_size": args.batch_size,
        "prompt_length" if args.stage == "prefill" else "initial_kv_length": args.length,
        "decode_tokens": 1 if args.stage == "prefill" else args.decode_tokens,
        "dtype": args.dtype,
        "warmup_batches": args.warmup,
        "measured_batches": args.repeats,
        "submission_mode": args.submission_mode,
        "release_policy": (
            "all sequences submitted in one OpenAI batched-prompt request"
            if args.submission_mode == "batched_prompt" else
            "independent requests released in the same asyncio turn"
        ),
        "prompt_policy": (
            "exact-length token-id prompts with a request-unique first 16-token "
            "block to prevent prefix-cache reuse"),
        "metric": ("time until every request receives its first token"
                   if args.stage == "prefill" else
                   "mean per-request streamed TPOT"),
        "max_allowed_request_start_skew_ms": args.max_start_skew_ms,
        "max_allowed_first_token_spread_ms": args.max_first_token_spread_ms,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "collector": "vllm_fixed_batch_streaming",
        "experiment_type": "fixed_batch_iteration_validation",
        "model": args.model,
        "base_url": args.base_url,
        "configuration": benchmark,
        "deployment": deployment,
        "metadata": collect_runtime_metadata(
            model_path=args.tokenizer or args.model,
            benchmark=benchmark, project_root=PROJECT_ROOT),
        "serving_host_hardware": load_hardware_metadata(args),
        "warmup": warmups,
        "samples": samples,
        "summary": describe_ms(values) if values else None,
        "valid": transport_valid,
        "fixed_batch_valid": fixed_batch_valid,
        "validity": {
            "transport_valid": transport_valid,
            "fixed_batch_valid": fixed_batch_valid,
            "server_usage_matches_requested_batch": usage_valid,
            "first_token_cohort_valid": first_token_cohort_valid,
            "max_observed_first_token_spread_ms": max_first_token_spread,
            "scheduler_iteration_eligibility": eligibility,
        },
        "limitations": [
            "Prefill truth includes first-token sampling and serving overhead.",
            "Decode TPOT spans a growing KV length; --length is its initial KV length.",
            (
                "A batched prompt proves joint HTTP submission and scheduler-budget "
                "eligibility. The first-token cohort check rejects clearly serialized "
                "execution, but only an offline engine batch or server instrumentation "
                "can prove the exact internal batch boundary."
                if args.submission_mode == "batched_prompt" else
                "Concurrent client release cannot prove vLLM admitted every request "
                "into one batch."
            ),
        ],
    }
    write_profile(payload, args.output)
    print(
        f"saved fixed-batch run to {args.output}; "
        f"valid={payload['valid']}; fixed_batch_valid={fixed_batch_valid}")
    return 0 if (fixed_batch_valid or (
        args.submission_mode == "concurrent_requests" and transport_valid)) else 2


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
