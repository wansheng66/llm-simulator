#!/usr/bin/env python3
"""Measure TP collectives with NCCL or HCCL using a uniform schema."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.profiling_common import (  # noqa: E402
    SCHEMA_VERSION,
    collect_runtime_metadata,
    describe_ms,
    write_profile,
)
from scripts.accelerator_runtime import resolve_accelerator  # noqa: E402


def csv_floats(value: str) -> List[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive sizes")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message-sizes-mb", type=csv_floats,
                        default=csv_floats("0.01,0.02,0.05,0.1,0.2,0.5,1,2,4,8,16,32,64,128,256"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"),
                        default="bfloat16")
    parser.add_argument("--platform", choices=("auto", "cuda", "ascend"),
                        default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--keep-samples", action="store_true")
    return parser.parse_args()


def _one_accelerator_sample(runtime, function) -> float:
    start = runtime.event()
    end = runtime.event()
    start.record()
    function()
    end.record()
    runtime.synchronize()
    return float(start.elapsed_time(end))


def distributed_samples(torch, runtime, dist, function, warmup: int,
                        repeats: int) -> List[float]:
    for _ in range(warmup):
        dist.barrier()
        function()
        runtime.synchronize()
    samples = []
    for _ in range(repeats):
        dist.barrier()
        local_ms = _one_accelerator_sample(runtime, function)
        maximum = torch.tensor(
            local_ms, dtype=torch.float32,
            device=runtime.device(int(os.environ.get("LOCAL_RANK", 0))))
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        samples.append(float(maximum.item()))
    return samples


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("warmup must be non-negative and repeats must be positive")
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    try:
        runtime = resolve_accelerator(torch, args.platform)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    runtime.set_device(local_rank)
    device = runtime.device(local_rank)
    accelerator_device = torch.device(device)
    try:
        dist.init_process_group(
            backend=runtime.distributed_backend, device_id=accelerator_device)
    except TypeError:  # compatibility with PyTorch releases without device_id
        dist.init_process_group(backend=runtime.distributed_backend)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    p2p_group = (dist.new_group(
        ranks=[0, 1], backend=runtime.distributed_backend)
                 if world_size >= 2 else None)
    dtype = getattr(torch, args.dtype)
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    measurements: List[Dict] = []

    for size_mb in args.message_sizes_mb:
        elements = max(1, int(size_mb * 1024 * 1024 / bytes_per_element))
        # Zero is stable under repeated SUM all-reduce and avoids FP16 overflow
        # without adding a reset kernel to the timed region.
        source = torch.zeros(elements, dtype=dtype, device=device)
        all_gather_output = torch.empty(elements * world_size, dtype=dtype, device=device)
        reduce_scatter_input = torch.ones(elements * world_size, dtype=dtype, device=device)
        reduce_scatter_output = torch.empty(elements, dtype=dtype, device=device)

        operations = {
            "all_reduce": lambda: dist.all_reduce(source),
            "all_gather": lambda: dist.all_gather_into_tensor(all_gather_output, source),
            "reduce_scatter": lambda: dist.reduce_scatter_tensor(
                reduce_scatter_output, reduce_scatter_input),
        }
        if world_size >= 2:
            def p2p():
                if rank == 0:
                    dist.isend(source, dst=1, group=p2p_group).wait()
                elif rank == 1:
                    dist.irecv(source, src=0, group=p2p_group).wait()
            operations["p2p"] = p2p

        for operation, function in operations.items():
            samples = distributed_samples(
                torch, runtime, dist, function, args.warmup, args.repeats)
            if rank == 0:
                timing = describe_ms(samples)
                if args.keep_samples:
                    timing["samples_ms"] = samples
                measurements.append({
                    "status": "success",
                    "operation": operation,
                    "world_size": world_size,
                    "message_size_mb_per_rank": size_mb,
                    "dtype": args.dtype,
                    "timing": timing,
                })
                print(f"{operation} TP={world_size} {size_mb:g} MB: "
                      f"{timing['mean_ms']:.4f} ms")
        del source, all_gather_output, reduce_scatter_input, reduce_scatter_output
        runtime.empty_cache()

    if rank == 0:
        benchmark = {
            "kind": "tensor_parallel_collective_microbenchmark",
            "platform": runtime.platform,
            "backend": runtime.distributed_backend,
            "world_size": world_size,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "timer": runtime.timer_name + "; maximum across ranks",
            "message_size_semantics": "payload size per rank",
        }
        payload = {
            "schema_version": SCHEMA_VERSION,
            "profile_type": "collective",
            "metadata": collect_runtime_metadata(
                benchmark=benchmark, project_root=PROJECT_ROOT),
            "measurements": measurements,
        }
        output = args.output_dir / f"collective_profile_tp{world_size}.json"
        write_profile(payload, output)
        print(f"saved {len(measurements)} points to {output}")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
