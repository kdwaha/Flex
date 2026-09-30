import json
import tempfile
import unittest
from pathlib import Path

from tools.export_public_results import export, without_text_outputs


class PublicResultExportTests(unittest.TestCase):
    def test_removes_generated_text_but_keeps_metrics_and_spectrum(self):
        record = {"test_loss": 1.2, "predictions": ["private text"],
                  "references": ["reference"], "generation_records": [{}],
                  "hessian_spectrum": [{"nodes": [1.0], "weights": [1.0]}]}
        self.assertEqual(without_text_outputs(record), {
            "test_loss": 1.2,
            "hessian_spectrum": [{"nodes": [1.0], "weights": [1.0]}],
        })

    def test_refuses_unfinished_runs_and_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "source" / "method" / "run"
            run.mkdir(parents=True)
            (run / "diagnostics.json").write_text("[]")
            report = root / "source" / "report"
            report.mkdir()
            (report / "comparison.md").write_text("Example report")
            target = root / "published"
            with self.assertRaises(ValueError):
                export(root / "source", target)
            self.assertFalse(target.exists())
            (run / "metrics.json").write_text(json.dumps({"algorithm": "method"}))
            target.mkdir()
            with self.assertRaises(FileExistsError):
                export(root / "source", target)
