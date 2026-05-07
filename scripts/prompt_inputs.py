from __future__ import annotations

import numpy as np


def _normalize_active_label(label: int) -> float:
    # Be tolerant to app-side enums: any positive value is foreground.
    # Zero/negative values are background.
    return 1.0 if float(label) > 0.0 else 0.0


def build_prompt_arrays(
    tracking_points: list[tuple[int, int, int]],
    max_points: int,
    fixed_length: bool,
) -> tuple[np.ndarray, np.ndarray]:
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
