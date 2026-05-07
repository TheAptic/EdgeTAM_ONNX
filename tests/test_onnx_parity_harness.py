import unittest
from pathlib import Path
import tempfile

import numpy as np

from scripts.onnx_parity_harness import _foreground_mask_from_logits
from scripts.onnx_parity_harness import _load_points_file
from scripts.onnx_parity_harness import _scenarios_from_points


class OnnxParityHarnessTests(unittest.TestCase):
    def test_foreground_mask_uses_negative_logits_without_prompts(self):
        logits = np.array([[1.5, 0.0, -0.1, -2.0]], dtype=np.float32)
        mask = _foreground_mask_from_logits(logits)
        self.assertEqual(mask.tolist(), [[0, 0, 1, 1]])

    def test_foreground_mask_can_select_positive_logits_polarity(self):
        logits = np.array([[-1.0, 2.0]], dtype=np.float32)
        # Positive point at x=1 should be inside only for logits > 0 polarity.
        mask = _foreground_mask_from_logits(logits, active_points=[(1, 0, 1)])
        self.assertEqual(mask.tolist(), [[0, 1]])

    def test_load_points_file_parses_percentage_points(self):
        txt = "\n".join(
            [
                "EdgeTAM point negative @ (5.00%, 80.00%)",
                "EdgeTAM point positive @ (50.00%, 25.00%)",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "points.txt"
            p.write_text(txt)
            points = _load_points_file(str(p), size=100)
        self.assertEqual(points[0][2], 0)
        self.assertEqual(points[1][2], 1)
        self.assertEqual(points[0][:2], (5, 79))
        self.assertEqual(points[1][:2], (50, 25))

    def test_scenarios_from_points_includes_positive_only_and_mixed(self):
        points = [(10, 10, 1), (20, 20, 0), (30, 30, 1)]
        scenarios = _scenarios_from_points(points)
        self.assertEqual([s["name"] for s in scenarios], ["positive_only_progression", "mixed_file_order_progression"])
        self.assertEqual(scenarios[0]["points"], [(10, 10, 1), (30, 30, 1)])
        self.assertEqual(scenarios[1]["points"], points)

    def test_load_points_file_parses_x_norm_format(self):
        txt = "\n".join(
            [
                "image_path=/tmp/ref.png",
                "image_width=2048",
                "image_height=1365",
                "prompt_count=2",
                "points:",
                "  1: label=1 x_norm=0.500000 y_norm=0.250000 x_px=1024 y_px=341",
                "  2: label=0 x_norm=0.100000 y_norm=0.900000 x_px=205 y_px=1228",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "points_norm.txt"
            p.write_text(txt)
            points = _load_points_file(str(p), size=100)
        self.assertEqual(points[0], (50, 25, 1))
        self.assertEqual(points[1], (10, 89, 0))


if __name__ == "__main__":
    unittest.main()
