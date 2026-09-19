import json
import tempfile
import unittest
from pathlib import Path

from core.benchmark_protocol import load_benchmark_spec
from scripts.predict_fixed_batch_benchmark import predict
from scripts.validate_relative_runtime import truth_ratios


ROOT = Path(__file__).resolve().parents[1]
SPEC = (ROOT / "configs" / "benchmark_specs" /
        "qwen3_32b_fixed_batch_tp4_v1.json")


class FakeCostModel:
    enable_e2e_compensation = False
    profile_statistic = "p50"

    @staticmethod
    def _estimate(stage, batch, length, tp):
        assert tp == 4
        total = (batch * length / 100.0 if stage == "prefill"
                 else batch * (1.0 + length / 1024.0))
        return {
            "total_time_ms": total,
            "extrapolated": False,
            "profile_source": "test_profile",
            "communication_profile_source": "test_collective",
            "estimated_bottleneck": "ffn",
            "operator_breakdown": {"ffn_ms": total},
            "warnings": [],
        }

    def predict_prefill(self, batch, length, tp):
        return self._estimate("prefill", batch, length, tp)

    def predict_decode(self, batch, length, tp):
        return self._estimate("decode", batch, length, tp)


class RelativeRuntimePredictionTests(unittest.TestCase):
    def test_basic_grid_has_exactly_eighteen_unique_cases(self):
        spec = load_benchmark_spec(SPEC)
        cases = [case["case"] for stage in ("prefill", "decode")
                 for case in spec["workloads"][stage]]
        self.assertEqual(len(cases), 18)
        self.assertEqual(len(set(cases)), 18)
        self.assertEqual(spec["scope"]["coverage"], "basic_grid_complete")

    def test_prediction_emits_protocol_native_points(self):
        spec = load_benchmark_spec(SPEC)
        result = predict(spec, FakeCostModel(), "test_hardware", 4, SPEC)
        self.assertEqual(result["coverage"]["point_count"], 18)
        self.assertEqual(result["coverage"]["extrapolated_point_count"], 0)
        prefill = next(item for item in result["prefill"]
                       if item["case"] == "P_B4_L512")
        decode = next(item for item in result["decode"]
                      if item["case"] == "D_B4_KV512")
        self.assertAlmostEqual(prefill["throughput_tok_s"], 100000.0)
        self.assertAlmostEqual(
            decode["throughput_tok_s"],
            4 * 1000.0 / (4 * (1.0 + 512 / 1024.0)))
        self.assertFalse(
            result["prediction_policy"]["full_fixed_batch_ground_truth_used_as_input"])

    def test_truth_orientation_can_be_reversed(self):
        payload = {
            "reference": {"hardware_id": "l20"},
            "candidate": {"hardware_id": "ascend"},
            "points": [{
                "stage": "prefill", "batch_size": 1,
                "representative_length": 128,
                "candidate_speedup_vs_reference": 2.0,
            }, {
                "stage": "decode", "batch_size": 4,
                "representative_length": 512,
                "candidate_speedup_vs_reference": 0.5,
            }],
        }
        same = truth_ratios(payload, "l20", "ascend")
        reverse = truth_ratios(payload, "ascend", "l20")
        self.assertEqual(same["P_B1_L128"], 2.0)
        self.assertEqual(same["D_B4_KV512"], 0.5)
        self.assertEqual(reverse["P_B1_L128"], 0.5)
        self.assertEqual(reverse["D_B4_KV512"], 2.0)


if __name__ == "__main__":
    unittest.main()
