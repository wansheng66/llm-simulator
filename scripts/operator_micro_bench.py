#!/usr/bin/env python3
"""Profile Qwen-style tensor-parallel Transformer operators on CUDA or Ascend.

This benchmark measures sharded compute only. Collective communication is kept
in ``collective_micro_bench.py`` so the cost model can compose and attribute the
two sources independently. Tensors and weights are allocated outside timed
regions; every point has warmup runs and a distribution of CUDA-event samples.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.profiling_common import (  # noqa: E402
    SCHEMA_VERSION,
    collect_runtime_metadata,
    describe_ms,
    dominant_component,
    write_profile,
)
from scripts.accelerator_runtime import (  # noqa: E402
    is_out_of_memory,
    resolve_accelerator,
)


DEFAULT_MODEL = {
    "num_hidden_layers": 64,
    "hidden_size": 5120,
    "intermediate_size": 25600,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
}


def csv_ints(value: str) -> List[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Model directory containing config.json")
    parser.add_argument("--stage", choices=("prefill", "decode", "both"), default="both")
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--batch-sizes", type=csv_ints, default=csv_ints("1,2,4,8,16,32"))
    parser.add_argument("--prefill-lengths", type=csv_ints,
                        default=csv_ints("128,512,1024,2048,4096"))
    parser.add_argument("--decode-kv-lengths", type=csv_ints,
                        default=csv_ints("128,512,1024,2048,4096"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--platform", choices=("auto", "cuda", "ascend"),
                        default="auto")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-samples", action="store_true")
    return parser.parse_args()


def load_model_config(model: str | None) -> Tuple[Dict, str]:
    config = dict(DEFAULT_MODEL)
    source = "built_in_qwen3_32b_defaults"
    if model:
        path = Path(model).expanduser() / "config.json"
        loaded = json.loads(path.read_text(encoding="utf-8"))
        for key in config:
            if loaded.get(key) is not None:
                config[key] = int(loaded[key])
        if loaded.get("head_dim") is None:
            hidden = int(config["hidden_size"])
            heads = int(config["num_attention_heads"])
            config["head_dim"] = hidden // heads
        source = str(path.resolve())
    return config, source


def benchmark_accelerator(runtime, function, warmup: int,
                          repeats: int) -> List[float]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    for _ in range(warmup):
        function()
    runtime.synchronize()
    samples = []
    for _ in range(repeats):
        start = runtime.event()
        end = runtime.event()
        start.record()
        function()
        end.record()
        runtime.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def _stats(samples: Sequence[float], keep_samples: bool) -> Dict:
    result = describe_ms(samples)
    if keep_samples:
        result["samples_ms"] = list(samples)
    return result


def _sdpa(torch, q, k, v, causal: bool):
    try:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal, enable_gqa=(q.shape[1] != k.shape[1]))
    except (TypeError, RuntimeError, NotImplementedError):
        # Older PyTorch and some torch_npu releases do not expose enable_gqa.
        # Repeating K/V is semantically equivalent and keeps the case usable.
        repeat = q.shape[1] // k.shape[1]
        return torch.nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(repeat, dim=1),
            v.repeat_interleave(repeat, dim=1), is_causal=causal)


def profile_shape(torch, runtime, stage: str, batch: int, length: int, tp_size: int,
                  model: Dict, dtype, device: str, warmup: int, repeats: int,
                  keep_samples: bool) -> Dict:
    hidden = model["hidden_size"]
    intermediate = model["intermediate_size"]
    q_heads = model["num_attention_heads"]
    kv_heads = model["num_key_value_heads"]
    head_dim = model["head_dim"]
    if q_heads % tp_size or kv_heads % tp_size or intermediate % tp_size:
        raise ValueError("attention heads, KV heads and intermediate size must divide TP")
    local_q_heads, local_kv_heads = q_heads // tp_size, kv_heads // tp_size
    local_intermediate = intermediate // tp_size
    tokens = batch * length if stage == "prefill" else batch

    x = torch.randn(tokens, hidden, dtype=dtype, device=device)
    norm_weight = torch.ones(hidden, dtype=dtype, device=device)
    qkv_width = (local_q_heads + 2 * local_kv_heads) * head_dim
    w_qkv = torch.randn(hidden, qkv_width, dtype=dtype, device=device)
    w_o = torch.randn(local_q_heads * head_dim, hidden, dtype=dtype, device=device)
    w_gate = torch.randn(hidden, local_intermediate, dtype=dtype, device=device)
    w_up = torch.randn(hidden, local_intermediate, dtype=dtype, device=device)
    w_down = torch.randn(local_intermediate, hidden, dtype=dtype, device=device)
    gate = torch.randn(tokens, local_intermediate, dtype=dtype, device=device)
    up = torch.randn(tokens, local_intermediate, dtype=dtype, device=device)
    down_input = torch.randn(tokens, local_intermediate, dtype=dtype, device=device)

    if stage == "prefill":
        q = torch.randn(batch, local_q_heads, length, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch, local_kv_heads, length, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch, local_kv_heads, length, head_dim, dtype=dtype, device=device)
        causal = True
    else:
        q = torch.randn(batch, local_q_heads, 1, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch, local_kv_heads, length, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch, local_kv_heads, length, head_dim, dtype=dtype, device=device)
        causal = False
    attention_output = torch.randn(tokens, local_q_heads * head_dim,
                                   dtype=dtype, device=device)
    residual_left = torch.randn(tokens, hidden, dtype=dtype, device=device)
    residual_right = torch.randn(tokens, hidden, dtype=dtype, device=device)

    operations = {
        "attention_rmsnorm": lambda: torch.nn.functional.rms_norm(
            x, (hidden,), norm_weight),
        "qkv_projection": lambda: torch.matmul(x, w_qkv),
        "attention": lambda: _sdpa(torch, q, k, v, causal),
        "output_projection": lambda: torch.matmul(attention_output, w_o),
        "attention_residual": lambda: residual_left + residual_right,
        "ffn_rmsnorm": lambda: torch.nn.functional.rms_norm(
            x, (hidden,), norm_weight),
        "ffn_gate_projection": lambda: torch.matmul(x, w_gate),
        "ffn_up_projection": lambda: torch.matmul(x, w_up),
        "ffn_activation": lambda: torch.nn.functional.silu(gate) * up,
        "ffn_down_projection": lambda: torch.matmul(down_input, w_down),
        "ffn_residual": lambda: residual_left + residual_right,
    }
    measured = {
        name: _stats(benchmark_accelerator(
            runtime, function, warmup, repeats), keep_samples)
        for name, function in operations.items()
    }
    attention_ms = sum(measured[name]["mean_ms"] for name in
                       ("attention_rmsnorm", "qkv_projection", "attention",
                        "output_projection", "attention_residual"))
    ffn_ms = sum(measured[name]["mean_ms"] for name in
                 ("ffn_rmsnorm", "ffn_gate_projection", "ffn_up_projection",
                  "ffn_activation", "ffn_down_projection"))
    ffn_ms += measured["ffn_residual"]["mean_ms"]
    components = {"attention": attention_ms, "ffn": ffn_ms}
    return {
        "status": "success",
        "stage": stage,
        "shape": {
            "batch_size": batch,
            "prompt_length" if stage == "prefill" else "kv_length": length,
            "tp_size": tp_size,
        },
        "operators": measured,
        "single_layer": {
            "attention_ms": attention_ms,
            "ffn_ms": ffn_ms,
            "total_compute_ms": attention_ms + ffn_ms,
            "bottleneck": dominant_component(components),
            "component_ratio": {
                key: value / (attention_ms + ffn_ms) for key, value in components.items()
            },
        },
    }


def main() -> int:
    args = parse_args()
    if args.tp_size <= 0:
        raise SystemExit("--tp-size must be positive")
    import torch

    try:
        runtime = resolve_accelerator(torch, args.platform)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    runtime.set_device(args.device_index)
    device = runtime.device(args.device_index)
    dtype = getattr(torch, args.dtype)
    model, config_source = load_model_config(args.model)
    stages = ("prefill", "decode") if args.stage == "both" else (args.stage,)
    measurements = []
    for stage in stages:
        lengths = args.prefill_lengths if stage == "prefill" else args.decode_kv_lengths
        for batch in args.batch_sizes:
            for length in lengths:
                print(f"profile {stage}: TP={args.tp_size}, batch={batch}, length={length}")
                try:
                    row = profile_shape(
                        torch, runtime, stage, batch, length, args.tp_size,
                        model, dtype, device, args.warmup, args.repeats,
                        args.keep_samples)
                except Exception as exc:
                    if not is_out_of_memory(exc):
                        raise
                    row = {
                        "status": "oom", "stage": stage,
                        "shape": {
                            "batch_size": batch,
                            "prompt_length" if stage == "prefill" else "kv_length": length,
                            "tp_size": args.tp_size,
                        },
                        "error": str(exc),
                    }
                measurements.append(row)
                gc.collect()
                runtime.empty_cache()
    benchmark = {
        "kind": "qwen_tensor_parallel_operator_microbenchmark",
        "stage": args.stage,
        "tp_size": args.tp_size,
        "dtype": args.dtype,
        "platform": runtime.platform,
        "device": device,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "model_config_source": config_source,
        "model_config": model,
        "timer": runtime.timer_name,
        "weights_allocated_outside_timed_region": True,
        "collectives_included": False,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "profile_type": "operator",
        "metadata": collect_runtime_metadata(
            model_path=args.model, benchmark=benchmark, project_root=PROJECT_ROOT),
        "measurements": measurements,
    }
    output = args.output_dir / f"operator_profile_tp{args.tp_size}.json"
    write_profile(payload, output)
    successful = sum(row["status"] == "success" for row in measurements)
    print(f"saved {successful}/{len(measurements)} successful points to {output}")
    return 0 if successful else 2


if __name__ == "__main__":
    raise SystemExit(main())
