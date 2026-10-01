import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class _DummyFlask:
    def route(self, *_args, **_kwargs):
        return lambda function: function

    def run(self, *_args, **_kwargs):
        return None


flask = types.ModuleType("flask")
flask.Flask = lambda *_args, **_kwargs: _DummyFlask()
flask.jsonify = lambda value: value
flask.send_from_directory = lambda *_args, **_kwargs: None
flask.request = types.SimpleNamespace(args={}, get_json=lambda: {})
flask_cors = types.ModuleType("flask_cors")
flask_cors.CORS = lambda *_args, **_kwargs: None
task_manager_module = types.ModuleType("core.task_manager")
task_manager_module.task_manager = types.SimpleNamespace()
with patch.dict(sys.modules, {
        "flask": flask,
        "flask_cors": flask_cors,
        "core.task_manager": task_manager_module}):
    import web.app as web_app


def prediction(hardware_id):
    return {
        "experiment_type": "fixed_batch_runtime_prediction",
        "meta": {"gpu_type": hardware_id, "tp": 4},
        "prefill": [{"case": "P_B1_L128", "throughput_tok_s": 10.0}],
        "decode": [{"case": "D_B1_KV128", "throughput_tok_s": 20.0}],
    }


def validation(reference_path, candidate_path, passed=True):
    return {
        "experiment_type": "fixed_batch_relative_runtime_validation",
        "status": "verified",
        "acceptance": {"passed": passed},
        "reference": {"gpu_type": "l20", "tp": 4},
        "candidate": {"gpu_type": "a3", "tp": 4},
        "scores": {
            "prefill_speedup": 2.0,
            "decode_speedup": 0.5,
            "paired_prefill_cases": 1,
            "paired_decode_cases": 1,
        },
        "validation": {
            "ground_truth_complete": True,
            "relative_mape_pct": 10.0,
            "rank_agreement_ratio": 1.0,
            "ground_truth_prefill_speedup": 1.8,
            "ground_truth_decode_speedup": 0.51,
            "prefill_score_error_pct": 11.11,
            "decode_score_error_pct": 1.96,
        },
        "points": [{
            "stage": "prefill",
            "case": "P_B1_L128",
            "candidate_throughput_tok_s": 20.0,
            "reference_throughput_tok_s": 10.0,
            "speedup": 2.0,
            "ground_truth_speedup": 1.8,
            "relative_error_pct": 11.11,
        }],
        "sources": {
            "reference_prediction": str(reference_path),
            "candidate_prediction": str(candidate_path),
        },
    }


class WebRuntimeResultsTests(unittest.TestCase):
    def test_summary_only_uses_predictions_with_passed_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "l20.json"
            candidate_path = root / "a3.json"
            reference_path.write_text(
                json.dumps(prediction("l20")), encoding="utf-8")
            candidate_path.write_text(
                json.dumps(prediction("a3")), encoding="utf-8")
            report_path = root / "relative_runtime_validation.json"
            report_path.write_text(json.dumps(validation(
                reference_path, candidate_path)), encoding="utf-8")
            with patch.object(
                    web_app, "RUNTIME_VALIDATION_PATTERN",
                    str(root / "**" / "relative_runtime_validation.json")):
                rows = web_app._runtime_prediction_summary()
            self.assertEqual({row["gpu"] for row in rows}, {"l20", "a3"})
            self.assertTrue(all(
                row["data_source"] == "validated_runtime_prediction"
                for row in rows))

            report_path.write_text(json.dumps(validation(
                reference_path, candidate_path, passed=False)),
                encoding="utf-8")
            with patch.object(
                    web_app, "RUNTIME_VALIDATION_PATTERN",
                    str(root / "**" / "relative_runtime_validation.json")):
                self.assertEqual(web_app._runtime_prediction_summary(), [])

    def test_relative_result_supports_reverse_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_path = root / "l20.json"
            candidate_path = root / "a3.json"
            report_path = root / "relative_runtime_validation.json"
            report_path.write_text(json.dumps(validation(
                reference_path, candidate_path)), encoding="utf-8")
            with patch.object(
                    web_app, "RUNTIME_VALIDATION_PATTERN",
                    str(root / "**" / "relative_runtime_validation.json")):
                result = web_app._runtime_validation_result("l20", "a3", 4)
            self.assertAlmostEqual(result["scores"]["prefill_speedup"], 0.5)
            self.assertTrue(result["validation"]["passed"])
            self.assertEqual(
                result["score_kind"], "calibrated_runtime_prediction")


if __name__ == "__main__":
    unittest.main()
