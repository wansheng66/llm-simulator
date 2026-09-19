"""Profile-driven fixed-batch cost model.

Measurements, interpolation, analytical fallback and optional end-to-end
calibration are deliberately separated. Every prediction reports provenance and
whether it left the measured profiling range.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from .memory_estimator import MemoryEstimator
    from .operator_interpolation import group_operator_times, interpolate_operator_table
    from .schema import CostEstimate, HardwareSpec, ModelSpec
except ImportError:  # direct script compatibility
    from memory_estimator import MemoryEstimator
    from operator_interpolation import group_operator_times, interpolate_operator_table
    from schema import CostEstimate, HardwareSpec, ModelSpec


QWEN3_32B = ModelSpec(
    name="Qwen3-32B", num_layers=64, hidden_size=5120,
    intermediate_size=25600, num_attention_heads=64, num_kv_heads=8,
    head_dim=128, num_parameters=32_000_000_000,
)


class LLMCostModel:
    """Predict iteration cost from operator and collective profiling tables.

    ``calibration_file`` can contain measured factors such as
    ``{"prefill": {"1": 1.12}, "decode": {"1": 1.08}}``. Unlike the old
    implementation, no hardware-specific correction factor is invented in code.
    """

    def __init__(self, data_dir: str, peak_tflops: float = 119.5,
                 mem_bw_gb_s: float = 864.0, model: ModelSpec = QWEN3_32B,
                 hardware: Optional[HardwareSpec] = None,
                 calibration_file: Optional[str] = None,
                 enable_e2e_compensation: bool = False,
                 operator_profile_dir: Optional[str] = None,
                 collective_profile_dir: Optional[str] = None,
                 profile_statistic: str = "mean", **legacy_kwargs):
        self.data_dir = Path(data_dir)
        self.model = model
        self.hardware = hardware or HardwareSpec(
            accelerator="unknown", peak_tflops=peak_tflops,
            memory_bandwidth_gb_s=mem_bw_gb_s)
        self.peak_tflops = peak_tflops
        self.mem_bw_gb_s = mem_bw_gb_s
        self.enable_e2e_compensation = enable_e2e_compensation
        if profile_statistic not in {"mean", "p50"}:
            raise ValueError("profile_statistic must be 'mean' or 'p50'")
        self.profile_statistic = profile_statistic
        timing_key = f"{profile_statistic}_ms"
        self.calibration = self._load_calibration(calibration_file)
        self.compute_table = self._load_required("qwen3_32b_prefill_lookup_table.json")
        self.decode_table = self._load_required("qwen3_32b_decode_lookup_table.json")
        self.operator_profile_tables = self._load_operator_profiles(
            Path(operator_profile_dir) if operator_profile_dir else
            self.data_dir / "operator_profiles", timing_key)
        tables = {tp: self._load_optional(f"comm_lookup_table_tp{tp}.json")
                  for tp in (1, 2, 4, 8)}
        self.comm_tables = {key: value for key, value in tables.items() if value}
        measured_collectives = self._load_collective_profiles(
            Path(collective_profile_dir) if collective_profile_dir else
            self.data_dir / "collective_profiles", timing_key)
        self.comm_tables.update(measured_collectives)
        self.collective_profile_tps = set(measured_collectives)
        self.num_layers = model.num_layers
        self.hidden_size = model.hidden_size
        self.intermediate_size = model.intermediate_size
        self.bytes_per_elem = model.bytes_per_element
        self.memory_estimator = MemoryEstimator(
            num_layers=model.num_layers, hidden_size=model.hidden_size,
            num_kv_heads=model.num_kv_heads, head_dim=model.head_dim,
            num_params=model.num_parameters, bytes_per_param=model.bytes_per_element)

    def _load_required(self, name: str) -> List[Dict]:
        path = self.data_dir / name
        if not path.exists():
            raise FileNotFoundError(f"profiling table not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list) or not data:
            raise ValueError(f"profiling table must be a non-empty JSON list: {path}")
        return data

    def _load_optional(self, name: str) -> List[Dict]:
        path = self.data_dir / name
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []

    @staticmethod
    def _load_operator_profiles(
        directory: Path, timing_key: str = "mean_ms",
    ) -> Dict[int, Dict[str, List[Dict]]]:
        """Load schema-v2 sharded compute profiles, retaining legacy fallback."""
        result: Dict[int, Dict[str, List[Dict]]] = {}
        if not directory.exists():
            return result
        for path in sorted(directory.glob("**/operator_profile_tp*.json")):
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("schema_version") != 2 or payload.get("profile_type") != "operator":
                raise ValueError(f"unsupported operator profile schema: {path}")
            benchmark = payload.get("metadata", {}).get("benchmark", {})
            tp_size = int(benchmark.get("tp_size", 0))
            if tp_size <= 0:
                raise ValueError(f"operator profile is missing tp_size: {path}")
            stage_tables = result.setdefault(tp_size, {"prefill": [], "decode": []})
            for measurement in payload.get("measurements", []):
                if measurement.get("status") != "success":
                    continue
                stage = measurement.get("stage")
                if stage not in stage_tables:
                    continue
                shape = measurement["shape"]
                layer = measurement["single_layer"]
                row = {"batch_size": int(shape["batch_size"])}
                for operator, timing in measurement.get("operators", {}).items():
                    if isinstance(timing, dict):
                        value = timing.get(timing_key, timing.get("mean_ms"))
                        if value is not None:
                            row[f"operator::{operator}"] = float(value)
                if stage == "prefill":
                    row.update({
                        "prompt_length": int(shape["prompt_length"]),
                        "ffn_gemm_ms": float(layer["ffn_ms"]),
                        "attention_prefill_ms": float(layer["attention_ms"]),
                    })
                else:
                    row.update({
                        "kv_length": int(shape["kv_length"]),
                        "ffn_gemv_ms": float(layer["ffn_ms"]),
                        "attention_decode_ms": float(layer["attention_ms"]),
                    })
                stage_tables[stage].append(row)
        return {
            tp: stages for tp, stages in result.items()
            if stages["prefill"] or stages["decode"]
        }

    @staticmethod
    def _load_collective_profiles(
        directory: Path, timing_key: str = "mean_ms",
    ) -> Dict[int, List[Dict]]:
        result: Dict[int, List[Dict]] = {}
        if not directory.exists():
            return result
        for path in sorted(directory.glob("**/collective_profile_tp*.json")):
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("schema_version") != 2 or payload.get("profile_type") != "collective":
                raise ValueError(f"unsupported collective profile schema: {path}")
            for measurement in payload.get("measurements", []):
                if (measurement.get("status") != "success" or
                        measurement.get("operation") != "all_reduce"):
                    continue
                tp_size = int(measurement["world_size"])
                timing = measurement["timing"]
                value = timing.get(timing_key, timing.get("mean_ms"))
                if value is None:
                    raise ValueError(
                        f"collective timing is missing {timing_key}: {path}")
                result.setdefault(tp_size, []).append({
                    "world_size": tp_size,
                    "msg_size_mb": float(measurement["message_size_mb_per_rank"]),
                    "allreduce_ms": float(value),
                    "timing_statistic": timing_key,
                })
        return result

    @staticmethod
    def _load_calibration(path: Optional[str]) -> Dict:
        if not path:
            return {}
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("calibration file must contain a JSON object")
        return data

    @staticmethod
    def _axis_bounds(values: Sequence[float], query: float) -> Tuple[float, float]:
        values = sorted(set(values))
        if len(values) == 1:
            return values[0], values[0]
        if query <= values[0]:
            return values[0], values[1]
        if query >= values[-1]:
            return values[-2], values[-1]
        index = bisect_right(values, query)
        return values[index - 1], values[index]

    @staticmethod
    def _log_fraction(low: float, high: float, value: float) -> float:
        if low == high:
            return 0.0
        low, high, value = max(low, 1e-9), max(high, 1e-9), max(value, 1e-9)
        return (math.log(value) - math.log(low)) / (math.log(high) - math.log(low))

    def _interpolate_2d(self, table: List[Dict], x_key: str, y_key: str,
                        value_key: str, x: float, y: float) -> Tuple[float, bool]:
        """Log-bilinear interpolation with bounded edge-slope extrapolation."""
        xs = sorted({float(row[x_key]) for row in table})
        ys = sorted({float(row[y_key]) for row in table})
        if not xs or not ys:
            raise ValueError("empty profiling grid")
        extrapolated = x < xs[0] or x > xs[-1] or y < ys[0] or y > ys[-1]
        x0, x1 = self._axis_bounds(xs, x)
        y0, y1 = self._axis_bounds(ys, y)
        grid = {(float(row[x_key]), float(row[y_key])): float(row[value_key])
                for row in table}

        def point(xx: float, yy: float) -> float:
            if (xx, yy) in grid:
                return grid[(xx, yy)]
            nearest = min(table, key=lambda row:
                          abs(math.log(float(row[x_key]) / xx)) +
                          abs(math.log(float(row[y_key]) / yy)))
            return float(nearest[value_key])

        fx = min(max(self._log_fraction(x0, x1, x), -1.0), 2.0)
        fy = min(max(self._log_fraction(y0, y1, y), -1.0), 2.0)
        v00, v10 = point(x0, y0), point(x1, y0)
        v01, v11 = point(x0, y1), point(x1, y1)
        lower = v00 + (v10 - v00) * fx
        upper = v01 + (v11 - v01) * fx
        return max(lower + (upper - lower) * fy, 0.001), extrapolated

    def _comm_time(self, tp_size: int, message_mb: float) -> Tuple[float, bool]:
        if tp_size <= 1:
            return 0.0, False
        table = self.comm_tables.get(tp_size)
        if not table:
            # Ring all-reduce fallback: alpha + 2*(N-1)/N*bytes/bandwidth.
            bandwidth = self.hardware.interconnect_bandwidth_gb_s or 25.0
            return 0.02 + 2 * (tp_size - 1) / tp_size * message_mb / bandwidth, True
        points = sorted((float(row["msg_size_mb"]), float(row["allreduce_ms"]))
                        for row in table)
        xs = [point[0] for point in points]
        low, high = self._axis_bounds(xs, message_mb)
        fraction = min(max(self._log_fraction(low, high, message_mb), -1.0), 2.0)
        values = dict(points)
        result = values[low] + (values[high] - values[low]) * fraction
        return max(result, 0.001), message_mb < xs[0] or message_mb > xs[-1]

    def _calibration_factor(self, stage: str, tp_size: int,
                            batch_size: int = 1, length: int = 1) -> float:
        """Return a backward-compatible scalar or shape-aware factor."""
        if not self.enable_e2e_compensation:
            return 1.0
        entry = self.calibration.get(stage, {}).get(str(tp_size), 1.0)
        if isinstance(entry, (int, float)):
            return float(entry)
        if not isinstance(entry, dict):
            raise ValueError(
                f"calibration {stage}.TP{tp_size} must be a number or object")
        factor = float(entry.get("base_factor", entry.get("factor", 1.0)))
        if stage == "decode":
            reference_batch = max(float(entry.get("reference_batch_size", 8.0)), 1.0)
            reference_length = max(float(entry.get("reference_kv_len", 1024.0)), 1.0)
            effective_batch = max(float(batch_size), 1.0)
            effective_length = max(float(length), 1.0)
            if entry.get("batch_size_min") is not None:
                effective_batch = max(
                    effective_batch, float(entry["batch_size_min"]))
            if entry.get("batch_size_max") is not None:
                effective_batch = min(
                    effective_batch, float(entry["batch_size_max"]))
            if entry.get("kv_len_min") is not None:
                effective_length = max(
                    effective_length, float(entry["kv_len_min"]))
            if entry.get("kv_len_max") is not None:
                effective_length = min(
                    effective_length, float(entry["kv_len_max"]))
            factor *= (effective_batch / reference_batch) ** float(
                entry.get("batch_exponent", 0.0))
            factor *= (effective_length / reference_length) ** float(
                entry.get("kv_exponent", 0.0))
        lower = float(entry.get("min_factor", 0.1))
        upper = float(entry.get("max_factor", 3.0))
        if not 0 < lower <= upper:
            raise ValueError("calibration factor bounds must satisfy 0 < min <= max")
        return min(max(factor, lower), upper)

    def _iteration_overhead_ms(self, tp_size: int) -> float:
        if not self.enable_e2e_compensation:
            return 0.0
        value = self.calibration.get("iteration_overhead_ms", {}).get(
            str(tp_size), 0.0)
        if isinstance(value, dict):
            value = value.get("value", 0.0)
        return max(float(value), 0.0)

    def _estimate(self, stage: str, batch_size: int, length: int, tp_size: int,
                  lengths: Optional[Sequence[int]] = None,
                  context_lengths: Optional[Sequence[int]] = None) -> Dict:
        if batch_size <= 0 or length <= 0 or tp_size <= 0:
            raise ValueError("batch_size, sequence length and tp_size must be positive")
        if stage == "prefill":
            table, y_key = self.compute_table, "prompt_length"
            ffn_key, attention_key = "ffn_gemm_ms", "attention_prefill_ms"
        else:
            table, y_key = self.decode_table, "kv_length"
            ffn_key, attention_key = "ffn_gemv_ms", "attention_decode_ms"
        profiled_table = self.operator_profile_tables.get(tp_size, {}).get(stage, [])
        profile_source = "tp_sharded_operator_profile" if profiled_table else "legacy_operator_profile"
        if profiled_table:
            table = profiled_table
        operator_detail = None
        if profiled_table and any(
                key.startswith("operator::") for row in profiled_table for key in row):
            operator_detail, operator_extra = interpolate_operator_table(
                profiled_table, stage, batch_size, length)
            grouped = group_operator_times(operator_detail)
            ffn, attention = grouped["ffn_ms"], grouped["attention_ms"]
            ffn_extra = attention_extra = operator_extra
        else:
            ffn, ffn_extra = self._interpolate_2d(
                table, "batch_size", y_key, ffn_key, batch_size, length)
            attention, attention_extra = self._interpolate_2d(
                table, "batch_size", y_key, attention_key, batch_size, length)
        if lengths and stage == "prefill":
            # Full prefill attention scales with sum(q^2). For a chunk with
            # existing context it scales approximately with sum(q * kv_end).
            if context_lengths:
                if len(context_lengths) != len(lengths):
                    raise ValueError("prefill chunks and contexts must align")
                attention_work = sum(float(query) * float(context)
                                     for query, context in zip(
                                         lengths, context_lengths))
            else:
                attention_work = sum(float(item) ** 2 for item in lengths)
            attention *= attention_work / max(
                batch_size * float(length) ** 2, 1.0)
        compute_ms = (ffn + attention) * self.num_layers
        if stage == "prefill":
            message_tokens = sum(lengths) if lengths else batch_size * length
        else:
            message_tokens = batch_size
        message_mb = message_tokens * self.hidden_size * self.bytes_per_elem / (1024 ** 2)
        comm_one, comm_extra = self._comm_time(tp_size, message_mb)
        communication_ms = comm_one * 2 * self.num_layers
        factor = self._calibration_factor(
            stage, tp_size, batch_size=batch_size, length=length)
        total_ms = (compute_ms + communication_ms) * factor
        memory = self.memory_estimator.estimate_total(
            batch_size=batch_size, seq_len=length, tp_size=tp_size,
            stage=stage, mode="inference")
        warnings = []
        if ffn_extra or attention_extra:
            warnings.append("compute shape is outside the measured profiling grid")
        if comm_extra:
            warnings.append("communication size/TP is outside the measured profiling grid")
        if factor != 1.0:
            warnings.append(f"applied measured end-to-end calibration factor {factor:.4f}")
        calibration_entry = self.calibration.get(stage, {}).get(str(tp_size), {})
        calibration_shape_clamped = False
        if self.enable_e2e_compensation and stage == "decode" and isinstance(
                calibration_entry, dict):
            calibration_shape_clamped = any((
                calibration_entry.get("batch_size_min") is not None and
                batch_size < float(calibration_entry["batch_size_min"]),
                calibration_entry.get("batch_size_max") is not None and
                batch_size > float(calibration_entry["batch_size_max"]),
                calibration_entry.get("kv_len_min") is not None and
                length < float(calibration_entry["kv_len_min"]),
                calibration_entry.get("kv_len_max") is not None and
                length > float(calibration_entry["kv_len_max"]),
            ))
            if calibration_shape_clamped:
                warnings.append(
                    "calibration residual was clamped to its measured shape support")
        result = CostEstimate(
            stage=stage, total_time_ms=total_ms, compute_ms=compute_ms * factor,
            communication_ms=communication_ms * factor, memory_mb=memory,
            extrapolated=ffn_extra or attention_extra or comm_extra,
            profile_source=profile_source + ("+e2e_calibration" if factor != 1 else ""),
            warnings=warnings).to_dict()
        result.update({"batch_size": batch_size, "tp_size": tp_size, y_key: length,
                       "calibration_factor": factor,
                       "calibration_shape_clamped": calibration_shape_clamped})
        result["communication_profile_source"] = (
            "schema_v2_collective_profile" if tp_size in self.collective_profile_tps
            else "legacy_collective_profile")
        result["profile_timing_statistic"] = self.profile_statistic
        result["seq_len" if stage == "prefill" else "kv_len"] = length
        operator_breakdown = {
            "attention_ms": attention * self.num_layers * factor,
            "ffn_ms": ffn * self.num_layers * factor,
            "collective_ms": communication_ms * factor,
            "runtime_overhead_ms": 0.0,
        }
        ranked = sorted(operator_breakdown.items(), key=lambda item: item[1], reverse=True)
        result["operator_breakdown"] = operator_breakdown
        if operator_detail is not None:
            result["operator_detail_breakdown"] = {
                name + "_ms": value * self.num_layers * factor
                for name, value in operator_detail.items()
            }
        else:
            result["operator_detail_breakdown"] = None
        result["estimated_bottleneck"] = (
            "balanced" if len(ranked) > 1 and ranked[0][1] < ranked[1][1] * 1.15
            else ranked[0][0].removesuffix("_ms"))
        return result

    def predict_prefill(self, batch_size: int, seq_len: int, tp_size: int) -> Dict:
        return self._estimate("prefill", batch_size, seq_len, tp_size)

    def predict_decode(self, batch_size: int, kv_len: int, tp_size: int) -> Dict:
        return self._estimate("decode", batch_size, kv_len, tp_size)

    def predict_batch(self, prefill_lengths: Sequence[int],
                      decode_kv_lengths: Sequence[int], tp_size: int,
                      prefill_context_lengths: Optional[Sequence[int]] = None) -> Dict:
        """Predict a continuous-batching iteration from actual request shapes."""
        parts = []
        if prefill_lengths:
            average = max(1, round(sum(prefill_lengths) / len(prefill_lengths)))
            parts.append(self._estimate("prefill", len(prefill_lengths), average,
                                        tp_size, prefill_lengths,
                                        prefill_context_lengths))
        if decode_kv_lengths:
            average = max(1, round(sum(decode_kv_lengths) / len(decode_kv_lengths)))
            parts.append(self._estimate("decode", len(decode_kv_lengths), average, tp_size))
        if not parts:
            raise ValueError("a batch must contain at least one request")
        iteration_overhead_ms = self._iteration_overhead_ms(tp_size)
        return {
            "stage": "mixed" if len(parts) == 2 else parts[0]["stage"],
            "total_time_ms": (sum(part["total_time_ms"] for part in parts) +
                              iteration_overhead_ms),
            "breakdown": {
                "compute_ms": sum(part["breakdown"]["compute_ms"] for part in parts),
                "comm_ms": sum(part["breakdown"]["comm_ms"] for part in parts),
                "iteration_overhead_ms": iteration_overhead_ms,
            },
            "extrapolated": any(part["extrapolated"] for part in parts),
            "warnings": [warning for part in parts for warning in part["warnings"]],
            "parts": parts,
        }


if __name__ == "__main__":
    data = Path(__file__).resolve().parents[1] / "data"
    print(json.dumps(LLMCostModel(str(data)).predict_prefill(4, 1024, 2),
                     indent=2, ensure_ascii=False))
