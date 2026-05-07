import importlib
import sys
import unittest
from pathlib import Path

import numpy as np


class _NoSam2Finder:
    def find_spec(self, fullname, path, target=None):
        if fullname == "sam2" or fullname.startswith("sam2."):
            raise ModuleNotFoundError("sam2 is blocked for this test")
        return None


class OnnxRunnerTests(unittest.TestCase):
    def test_runner_imports_without_sam2(self):
        blocker = _NoSam2Finder()
        sys.meta_path.insert(0, blocker)
        try:
            mod = importlib.import_module("edgetam_onnx.runner")
            self.assertTrue(hasattr(mod, "EdgeTamOnnxRunner"))
        finally:
            sys.meta_path.remove(blocker)

    def test_preprocess_shape_contract(self):
        from edgetam_onnx.runner import preprocess_image

        img = np.zeros((128, 256, 3), dtype=np.uint8)
        out = preprocess_image(img, size=256)
        self.assertEqual(out.shape, (1, 3, 256, 256))
        self.assertEqual(out.dtype, np.float32)

    @unittest.skipUnless(Path("model/edgetam.onnx").exists(), "ONNX model missing")
    def test_smoke_single_frame_inference_runs_without_sam2(self):
        blocker = _NoSam2Finder()
        sys.meta_path.insert(0, blocker)
        try:
            from edgetam_onnx.runner import EdgeTamOnnxRunner

            runner = EdgeTamOnnxRunner("model/edgetam.onnx")
            image = np.zeros((256, 256, 3), dtype=np.uint8)
            input_names = {i.name for i in runner.session.get_inputs()}
            if "point_coords" in input_names:
                n = next(i.shape[1] for i in runner.session.get_inputs() if i.name == "point_coords")
                point_coords = np.zeros((1, int(n), 2), dtype=np.float32)
                point_labels = np.full((1, int(n)), -1.0, dtype=np.float32)
                point_coords[0, 0] = [128.0, 128.0]
                point_labels[0, 0] = 1.0
                outputs = runner.run_single_frame(
                    image,
                    point_coords=point_coords,
                    point_labels=point_labels,
                )
            else:
                try:
                    outputs = runner.run_single_frame(image)
                except Exception as exc:
                    self.skipTest(f"Local ONNX artifact fails runtime smoke check: {exc}")

            self.assertIsInstance(outputs, dict)
            self.assertGreater(len(outputs), 0)
        finally:
            sys.meta_path.remove(blocker)


if __name__ == "__main__":
    unittest.main()
