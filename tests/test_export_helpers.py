import unittest
from pathlib import Path

import torch

from scripts.export_edgetam_onnx import _is_filesystem_config
from scripts.export_edgetam_onnx import _sanitize_point_labels_for_onnx


class ExportHelperTests(unittest.TestCase):
    def test_detects_filesystem_yaml_config(self):
        self.assertTrue(_is_filesystem_config("EdgeTAM/checkpoints/edgetam.yaml"))
        self.assertFalse(_is_filesystem_config(str(Path("/tmp/a.yaml"))))

    def test_detects_hydra_package_config(self):
        self.assertFalse(_is_filesystem_config("configs/sam2/sam2_hiera_l.yaml"))

    def test_sanitize_point_labels_for_onnx_contract(self):
        labels = torch.tensor([[1.0, 2.0, 0.0, -3.0, -1.0]])
        out = _sanitize_point_labels_for_onnx(labels, torch_module=torch)
        self.assertEqual(out.tolist(), [[1.0, 1.0, 0.0, 0.0, -1.0]])


if __name__ == "__main__":
    unittest.main()
