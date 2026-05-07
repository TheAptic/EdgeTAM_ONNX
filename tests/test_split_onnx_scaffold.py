import importlib
import unittest


class SplitOnnxScaffoldTests(unittest.TestCase):
    def test_runtime_modules_import(self):
        mod = importlib.import_module("edgetam_onnx.runtime.ort_session")
        self.assertTrue(hasattr(mod, "create_split_sessions"))

    def test_validate_module_import(self):
        mod = importlib.import_module("edgetam_onnx.validate.compare_pytorch_onnx")
        self.assertTrue(hasattr(mod, "main"))


if __name__ == "__main__":
    unittest.main()
