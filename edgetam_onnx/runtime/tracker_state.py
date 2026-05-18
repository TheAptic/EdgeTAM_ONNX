"""State container for frame-to-frame memory tensors outside ONNX graphs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackerState:
    """Mutable tracker memory passed explicitly by application code."""
    # Placeholder for explicit memory tensors carried frame-to-frame.
    # Keep tracker loop outside ONNX and update these in app code.
    memory_k: np.ndarray | None = None
    memory_v: np.ndarray | None = None
