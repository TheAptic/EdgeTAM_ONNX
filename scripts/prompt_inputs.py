"""Helpers for converting UI points into model-ready prompt tensors."""

from __future__ import annotations

import numpy as np


def _normalize_active_label(label: int) -> float:
    """Map app-side labels to the ONNX prompt contract.

    The exported prompt encoder expects `1` for foreground and `0` for
    background. Some callers use richer enums, so any positive value is treated
    as foreground and non-positive values are treated as background.
    """
    return 1.0 if float(label) > 0.0 else 0.0


def build_prompt_arrays(
    tracking_points: list[tuple[int, int, int]],
    max_points: int,
    fixed_length: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build `(point_coords, point_labels)` with batch dimension.

    Args:
        tracking_points: List of `(x, y, label)` tuples in pixel coordinates.
        max_points: Maximum number of prompt slots to export.
        fixed_length: When `True`, pad to `max_points` using label `-1` for
            unused slots to match static-shape exports.
    """
    if not tracking_points:
        raise ValueError("tracking_points must not be empty")

    if fixed_length:
        n = max_points
        coords = np.zeros((1, n, 2), dtype=np.float32)
        labels = np.full((1, n), -1, dtype=np.float32)
        for i, (x, y, label) in enumerate(tracking_points[:max_points]):
            coords[0, i] = [x, y]
            labels[0, i] = _normalize_active_label(label)
        return coords, labels

    n = min(len(tracking_points), max_points)
    coords = np.zeros((1, n, 2), dtype=np.float32)
    labels = np.zeros((1, n), dtype=np.float32)
    for i, (x, y, label) in enumerate(tracking_points[:n]):
        coords[0, i] = [x, y]
        labels[0, i] = _normalize_active_label(label)
    return coords, labels
