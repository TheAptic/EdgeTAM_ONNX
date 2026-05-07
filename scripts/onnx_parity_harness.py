from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image

from edgetam_onnx.runner import EdgeTamOnnxRunner
from scripts.prompt_inputs import build_prompt_arrays


def _overlay(image: np.ndarray, mask: np.ndarray, prompts: list[tuple[int, int, int]]) -> np.ndarray:
    out = image.copy()
    if mask.ndim == 3:
        mask = mask.squeeze()
    mask = (mask > 0).astype(np.uint8)
    tint = np.zeros_like(out)
    tint[:, :, 1] = 255
    alpha = 0.35
    out = np.where(mask[..., None] > 0, (out * (1 - alpha) + tint * alpha).astype(np.uint8), out)
    for x, y, label in prompts:
        color = (0, 255, 0) if label > 0 else (0, 0, 255)
        cv2.circle(out, (int(x), int(y)), 10, color, thickness=2)
    return out


def _resize_image(image: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(image).resize((size, size), Image.BILINEAR), dtype=np.uint8)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = (a > 0).astype(np.uint8)
    bb = (b > 0).astype(np.uint8)
    inter = int((aa & bb).sum())
    union = int((aa | bb).sum())
    if union == 0:
        return 1.0
    return inter / union


def _tensor_stats(arr: np.ndarray) -> dict[str, Any]:
    a = np.asarray(arr, dtype=np.float32)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
    }


def _load_edge_mask(mask_path: str, size: int) -> np.ndarray:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read edge mask: {mask_path}")
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


_POINT_LINE_RE = re.compile(
    r"EdgeTAM point\s+(positive|negative)\s+@\s+\(([-+]?\d+(?:\.\d+)?)%,\s*([-+]?\d+(?:\.\d+)?)%\)",
    re.IGNORECASE,
)
_POINT_NORM_LINE_RE = re.compile(
    r"^\s*\d+\s*:\s*label\s*=\s*([-+]?\d+)\s+x_norm\s*=\s*([-+]?\d+(?:\.\d+)?)\s+y_norm\s*=\s*([-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _load_points_file(points_file: str, size: int) -> list[tuple[int, int, int]]:
    points: list[tuple[int, int, int]] = []
    content = Path(points_file).read_text().splitlines()
    for line in content:
        m = _POINT_LINE_RE.search(line.strip())
        if m:
            label_txt, x_pct_s, y_pct_s = m.groups()
            x = int(round((float(x_pct_s) / 100.0) * (size - 1)))
            y = int(round((float(y_pct_s) / 100.0) * (size - 1)))
            label = 1 if label_txt.lower() == "positive" else 0
            points.append((x, y, label))
            continue

        m_norm = _POINT_NORM_LINE_RE.search(line.strip())
        if m_norm:
            label_s, x_norm_s, y_norm_s = m_norm.groups()
            x = int(round(float(x_norm_s) * (size - 1)))
            y = int(round(float(y_norm_s) * (size - 1)))
            label = 1 if int(label_s) > 0 else 0
            points.append((x, y, label))

    if not points:
        raise ValueError(f"No valid points parsed from: {points_file}")
    return points


def _foreground_mask_from_logits(
    logits: np.ndarray,
    active_points: list[tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """
    Choose foreground polarity by prompt consistency.
    Falls back to logits < 0 when no points are available.
    """
    neg = (logits < 0).astype(np.uint8)
    pos = (logits > 0).astype(np.uint8)
    if not active_points:
        return neg

    def _score(mask: np.ndarray) -> int:
        s = 0
        for x, y, lbl in active_points:
            inside = _point_in_mask(mask, int(x), int(y))
            if lbl > 0:
                s += 1 if inside else -1
            else:
                s += 1 if not inside else -1
        return s

    score_neg = _score(neg)
    score_pos = _score(pos)
    if score_pos > score_neg:
        return pos
    if score_neg > score_pos:
        return neg
    return pos if int(pos.sum()) < int(neg.sum()) else neg


def _point_in_mask(mask: np.ndarray, x: int, y: int) -> int:
    h, w = mask.shape[-2:]
    if x < 0 or y < 0 or x >= w or y >= h:
        return 0
    return int(mask[y, x] > 0)


def _default_scenarios(size: int) -> list[dict[str, Any]]:
    c = size // 2
    return [
        {"name": "single_positive_center", "points": [(c, c, 1)]},
        {"name": "positive_positive", "points": [(c, c, 1), (c + 90, c - 20, 1)]},
        {"name": "positive_negative", "points": [(c, c, 1), (c + 120, c + 40, 0)]},
        {
            "name": "mixed_6_points",
            "points": [
                (c, c, 1),
                (c + 70, c - 30, 1),
                (c - 80, c + 20, 1),
                (c + 130, c + 80, 0),
                (c - 140, c - 90, 0),
                (c + 20, c + 150, 1),
            ],
        },
    ]


def _scenarios_from_points(points: list[tuple[int, int, int]]) -> list[dict[str, Any]]:
    positives = [p for p in points if p[2] > 0]
    scenarios: list[dict[str, Any]] = []
    if positives:
        scenarios.append({"name": "positive_only_progression", "points": positives})
    scenarios.append({"name": "mixed_file_order_progression", "points": points})
    return scenarios


def _build_torch_runner(config: str, checkpoint: str, sam2_root: Path):
    sys.path.insert(0, str(sam2_root))
    import torch
    from hydra import compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config) if Path(config).exists() else compose(config_name=config)
    model = instantiate(cfg.model, _recursive_=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(state, strict=False)
    model.eval()

    class _TorchRunner:
        def run(self, image_nchw: np.ndarray, point_coords: np.ndarray, point_labels: np.ndarray) -> np.ndarray:
            image = torch.from_numpy(image_nchw).float()
            coords = torch.from_numpy(point_coords).float()
            labels = torch.from_numpy(point_labels).float()
            with torch.no_grad():
                backbone_out = model.forward_image(image)
                _, vision_feats, vision_pos_embeds, feat_sizes = model._prepare_backbone_features(backbone_out)
                out = model.track_step(
                    frame_idx=0,
                    is_init_cond_frame=True,
                    current_vision_feats=vision_feats,
                    current_vision_pos_embeds=vision_pos_embeds,
                    feat_sizes=feat_sizes,
                    point_inputs={"point_coords": coords, "point_labels": labels},
                    mask_inputs=None,
                    output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
                    num_frames=1,
                    run_mem_encoder=False,
                )
                logits = out["pred_masks_high_res"][0, 0].cpu().numpy()
            return logits

    return _TorchRunner()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--out-dir", default="artifacts/onnx_parity")
    parser.add_argument("--out", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--max-points", type=int, default=4)
    parser.add_argument("--scenarios-json", default=None)
    parser.add_argument("--points-file", default=None)
    parser.add_argument("--edge-mask", default=None)
    parser.add_argument("--config", default="model/edgetam.yaml")
    parser.add_argument("--checkpoint", default="model/edgetam.pt")
    parser.add_argument("--sam2-root", default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    default_root = project_root / "EdgeTAM" if (project_root / "EdgeTAM").exists() else project_root / "sam2"
    sam2_root = Path(args.sam2_root) if args.sam2_root else default_root

    image = cv2.cvtColor(cv2.imread(args.image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    image = _resize_image(image, args.size)
    image_nchw = np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
    onnx_runner = EdgeTamOnnxRunner(args.onnx)
    torch_runner = _build_torch_runner(args.config, args.checkpoint, sam2_root)

    if args.points_file:
        scenarios = _scenarios_from_points(_load_points_file(args.points_file, args.size))
    elif args.scenarios_json:
        scenarios = json.loads(Path(args.scenarios_json).read_text())
    else:
        scenarios = _default_scenarios(args.size)

    edge_mask = _load_edge_mask(args.edge_mask, args.size) if args.edge_mask else None

    out_dir = Path(args.out_dir)
    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    contract: dict[str, Any] = {
        "image_path": args.image,
        "size": int(args.size),
        "max_points": int(args.max_points),
        "preprocess": {
            "resize": "PIL.BILINEAR",
            "input_color_space": "RGB",
            "tensor_layout": "NCHW",
            "tensor_range": "[0,1]",
            "image_uint8_stats": _tensor_stats(image),
            "image_nchw_stats": _tensor_stats(image_nchw),
        },
        "prompt_contract": {
            "fixed_length": True,
            "label_semantics": {"positive": 1.0, "negative": 0.0, "unused": -1.0},
            "points_source": args.points_file or args.scenarios_json or "built-in defaults",
        },
        "first_step_deltas": None,
    }

    for scenario in scenarios:
        active: list[tuple[int, int, int]] = []
        for step_idx, point in enumerate(scenario["points"], start=1):
            active.append(tuple(point))
            coords, labels = build_prompt_arrays(active, max_points=args.max_points, fixed_length=True)
            onnx_out = onnx_runner.run_single_frame(
                image,
                size=args.size,
                point_coords=coords,
                point_labels=labels,
            )
            onnx_logits = onnx_out["pred_masks_high_res"][0, 0]
            torch_logits = torch_runner.run(image_nchw, coords, labels)
            onnx_mask = _foreground_mask_from_logits(onnx_logits, active_points=active)
            torch_mask = _foreground_mask_from_logits(torch_logits, active_points=active)
            iou = _iou(onnx_mask, torch_mask)
            if contract["first_step_deltas"] is None:
                logits_delta = np.abs(onnx_logits.astype(np.float32) - torch_logits.astype(np.float32))
                contract["first_step_deltas"] = {
                    "scenario": scenario["name"],
                    "step": int(step_idx),
                    "point_count": len(active),
                    "coords_stats": _tensor_stats(coords),
                    "labels_stats": _tensor_stats(labels),
                    "onnx_logits_stats": _tensor_stats(onnx_logits),
                    "torch_logits_stats": _tensor_stats(torch_logits),
                    "abs_logits_delta": {
                        "mean": float(logits_delta.mean()),
                        "p95": float(np.percentile(logits_delta, 95)),
                        "max": float(logits_delta.max()),
                    },
                    "mask_delta": {
                        "onnx_area_px": int(onnx_mask.sum()),
                        "torch_area_px": int(torch_mask.sum()),
                        "iou_onnx_vs_torch": float(iou),
                    },
                }

            pos_ok = 0
            neg_ok = 0
            pos_total = 0
            neg_total = 0
            for x, y, lbl in active:
                pin = _point_in_mask(onnx_mask, int(x), int(y))
                if lbl > 0:
                    pos_total += 1
                    pos_ok += pin
                else:
                    neg_total += 1
                    neg_ok += int(pin == 0)

            scenario_name = scenario["name"]
            overlay = _overlay(image, onnx_mask, active)
            overlay_path = overlays_dir / f"{scenario_name}_step{step_idx:02d}.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            rows.append(
                {
                    "scenario": scenario_name,
                    "step": step_idx,
                    "active_points": len(active),
                    "iou_onnx_vs_torch": iou,
                    "iou_onnx_vs_reference_mask": _iou(onnx_mask, edge_mask) if edge_mask is not None else None,
                    "iou_torch_vs_reference_mask": _iou(torch_mask, edge_mask) if edge_mask is not None else None,
                    "onnx_area_px": int(onnx_mask.sum()),
                    "torch_area_px": int(torch_mask.sum()),
                    "area_ratio_onnx_over_torch": (float(onnx_mask.sum()) / float(torch_mask.sum()))
                    if int(torch_mask.sum()) > 0
                    else None,
                    "pos_inclusion_rate": (pos_ok / pos_total) if pos_total else 1.0,
                    "neg_suppression_rate": (neg_ok / neg_total) if neg_total else 1.0,
                    "overlay_path": str(overlay_path),
                }
            )

    csv_path = out_dir / "parity_report.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "rows": len(rows),
        "iou_median": float(np.median([r["iou_onnx_vs_torch"] for r in rows])),
        "iou_p10": float(np.percentile([r["iou_onnx_vs_torch"] for r in rows], 10)),
        "point_in_mask_agreement": float(np.mean([r["pos_inclusion_rate"] for r in rows])),
        "neg_suppression_agreement": float(np.mean([r["neg_suppression_rate"] for r in rows])),
        "csv_path": str(csv_path),
        "overlays_dir": str(overlays_dir),
    }
    if edge_mask is not None:
        onnx_vs_ref = [r["iou_onnx_vs_reference_mask"] for r in rows]
        torch_vs_ref = [r["iou_torch_vs_reference_mask"] for r in rows]
        summary["iou_onnx_vs_reference_median"] = float(np.median(onnx_vs_ref))
        summary["iou_torch_vs_reference_median"] = float(np.median(torch_vs_ref))
        summary["early_click_area_ratio_max"] = float(
            max(
                (
                    r["area_ratio_onnx_over_torch"]
                    for r in rows
                    if "positive_only" in r["scenario"] and int(r["step"]) <= 4 and r["area_ratio_onnx_over_torch"] is not None
                ),
                default=float("nan"),
            )
        )
    summary_path = Path(args.out) if args.out else (out_dir / "summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    contract_path = out_dir / "contract_report.json"
    contract_path.write_text(json.dumps(contract, indent=2))
    summary["contract_report_path"] = str(contract_path)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
