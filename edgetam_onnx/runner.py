from __future__ import annotations

from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image


def preprocess_image(image: np.ndarray, size: int = 1024) -> np.ndarray:
    """Convert HWC uint8 image into NCHW float32 tensor in [0, 1]."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected image shape (H, W, 3)")

    pil = Image.fromarray(image)
    resized = pil.resize((size, size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


class EdgeTamOnnxRunner:
    """Pure ONNX Runtime single-frame inference runner."""

    def __init__(self, model_path: str, providers: list[Any] | None = None):
        self.session = ort.InferenceSession(
            model_path,
            providers=providers or ["CPUExecutionProvider"],
        )

    def _session_input_names(self) -> set[str]:
        return {i.name for i in self.session.get_inputs()}

    def run_single_frame(
        self,
        image: np.ndarray,
        size: int = 1024,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
    ) -> dict[str, Any]:
        inp = preprocess_image(image, size=size)
        feed: dict[str, Any] = {"image": inp}
        input_names = self._session_input_names()
        if "point_coords" in input_names:
            if point_coords is None or point_labels is None:
                raise ValueError("point_coords and point_labels are required by this ONNX model")
            feed["point_coords"] = point_coords.astype(np.float32, copy=False)
            feed["point_labels"] = point_labels.astype(np.float32, copy=False)
        outputs = self.session.run(None, feed)
        output_names = [o.name for o in self.session.get_outputs()]
        return dict(zip(output_names, outputs, strict=True))
