"""Small CUDA/Ascend compatibility layer for profiling scripts.

The benchmark payload must stay backend-neutral.  This module contains the
only platform-specific event, synchronization and distributed-backend logic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AcceleratorRuntime:
    platform: str
    device_type: str
    distributed_backend: str
    module: object

    def device(self, index: int = 0) -> str:
        return f"{self.device_type}:{index}"

    def set_device(self, index: int) -> None:
        self.module.set_device(index)

    def synchronize(self) -> None:
        self.module.synchronize()

    def event(self):
        return self.module.Event(enable_timing=True)

    def empty_cache(self) -> None:
        self.module.empty_cache()

    @property
    def timer_name(self) -> str:
        return f"torch.{self.device_type}.Event"


def resolve_accelerator(torch, platform: str = "auto") -> AcceleratorRuntime:
    """Resolve CUDA or Ascend without changing the profiling schema."""
    if platform not in {"auto", "cuda", "ascend"}:
        raise ValueError(f"unsupported platform: {platform}")

    if platform in {"auto", "ascend"}:
        try:
            import torch_npu  # noqa: F401  # registers torch.npu
        except ImportError:
            if platform == "ascend":
                raise RuntimeError("--platform ascend requires torch_npu")
        npu = getattr(torch, "npu", None)
        if npu is not None and npu.is_available():
            return AcceleratorRuntime("ascend", "npu", "hccl", npu)
        if platform == "ascend":
            raise RuntimeError("Ascend NPU is not available")

    if platform in {"auto", "cuda"} and torch.cuda.is_available():
        return AcceleratorRuntime("cuda", "cuda", "nccl", torch.cuda)
    if platform == "cuda":
        raise RuntimeError("CUDA is not available")
    raise RuntimeError("neither Ascend NPU nor CUDA is available")


def is_out_of_memory(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    return "outofmemory" in name or "out of memory" in message or "oom" in message
