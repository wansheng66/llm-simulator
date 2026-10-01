import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.collect_fixed_batch_spec import (
    load_cases,
    stage_command,
    validate_stage_config,
)
from scripts.collect_fixed_batch_suite import exact_case, requested_points


ROOT = Path(__file__).resolve().parents[1]
SPEC = (
    ROOT / "configs" / "benchmark_specs" /
    "qwen3_32b_runtime_operator_extrapolation_4physical_v3.json"
)


class FixedBatchSpecTests(unittest.TestCase):
    def test_exact_cases_override_legacy_cartesian_grid(self):
        args = argparse.Namespace(
            exact_cases=[("prefill", 5, 1536), ("decode", 12, 512)],
            stages=["prefill", "decode"],
            batch_sizes=[1, 4, 8],
            prefill_lengths=[128, 512, 1024],
            decode_kv_lengths=[128, 512, 1024],
        )
        self.assertEqual(
            requested_points(args),
            [("prefill", 5, 1536), ("decode", 12, 512)],
        )

    def test_exact_case_parser_rejects_bad_shape(self):
        self.assertEqual(exact_case("prefill:4:2048"), ("prefill", 4, 2048))
        with self.assertRaises(argparse.ArgumentTypeError):
            exact_case("prefill:0:2048")
        with self.assertRaises(argparse.ArgumentTypeError):
            exact_case("unknown:4:2048")

    def test_loads_eight_cases_and_builds_four_case_stage_command(self):
        _, cases = load_cases(SPEC)
        prefill = [case for case in cases if case["stage"] == "prefill"]
        command = stage_command(
            ["--model", "/model", "--tp-size", "4"],
            prefill,
            "prefill",
            Path("prefill.json"),
            Path("output"),
        )
        self.assertEqual(len(cases), 8)
        self.assertEqual(command.count("--exact-case"), 4)
        self.assertIn("prefill:5:1536", command)
        self.assertIn("--offline-engine-lifecycle", command)
        self.assertIn("suite", command)

    def test_stage_specific_budget_preflight(self):
        spec, cases = load_cases(SPEC)
        decode = [case for case in cases if case["stage"] == "decode"]
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "decode.json"
            deployment = {
                "tensor_parallel_size": 4,
                "max_model_len": 8192,
                "max_num_batched_tokens": 16384,
                "max_num_seqs": 32,
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
            }
            config.write_text(json.dumps(deployment), encoding="utf-8")
            loaded = validate_stage_config(spec, decode, "decode", config, 4)
            self.assertEqual(loaded["max_num_batched_tokens"], 16384)

            deployment["max_num_batched_tokens"] = 8192
            config.write_text(json.dumps(deployment), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires 16384"):
                validate_stage_config(spec, decode, "decode", config, 4)


if __name__ == "__main__":
    unittest.main()
