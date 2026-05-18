"""Numerical parity checks between split ONNX outputs and PyTorch outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from edgetam_onnx.export.common import EdgeTAMImageEncoderExport
from edgetam_onnx.export.common import EdgeTAMMaskDecoderExport
from edgetam_onnx.export.common import EdgeTAMPromptEncoderExport
from edgetam_onnx.export.common import load_model
from edgetam_onnx.export.common import resolve_sam2_root
from edgetam_onnx.export.common import setup_sam2_import_path
from edgetam_onnx.runner import preprocess_image
from scripts.onnx_parity_harness import _load_points_file
from scripts.prompt_inputs import build_prompt_arrays


def _stats(a: np.ndarray) -> dict[str, Any]:
    """Return compact tensor statistics for diagnostics in parity reports."""
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
    }


def _compare(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> dict[str, Any]:
    """Compare tensors and return tolerance + error-distribution metadata."""
    diff = np.abs(a - b)
    ok = np.allclose(a, b, rtol=rtol, atol=atol)
    return {
        "allclose": bool(ok),
        "rtol": rtol,
        "atol": atol,
        "max_abs": float(np.max(diff)),
        "mean_abs": float(np.mean(diff)),
        "p99_abs": float(np.quantile(diff, 0.99)),
        "lhs": _stats(a),
        "rhs": _stats(b),
    }


def main() -> int:
    """Run parity report generation and return non-zero if any tensor fails."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--points-file", required=True)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--max-points", type=int, default=4)
    parser.add_argument("--config", default="model/edgetam.yaml")
    parser.add_argument("--checkpoint", default="model/edgetam.pt")
    parser.add_argument("--sam2-root", default=None)
    parser.add_argument("--image-encoder-onnx", default="edgetam_onnx/models/image_encoder.onnx")
    parser.add_argument("--prompt-encoder-onnx", default="edgetam_onnx/models/prompt_encoder.onnx")
    parser.add_argument("--mask-decoder-onnx", default="edgetam_onnx/models/mask_decoder.onnx")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--out", default="artifacts/onnx_split/compare_report.json")
    args = parser.parse_args()

    rtol, atol = (1e-3, 1e-3) if args.precision == "fp32" else (1e-2, 1e-2)

    sam2_root = resolve_sam2_root(args.sam2_root)
    setup_sam2_import_path(sam2_root)

    model = load_model(args.config, args.checkpoint)
    image_mod = EdgeTAMImageEncoderExport(model).module.eval()
    prompt_mod = EdgeTAMPromptEncoderExport(model).module.eval()
    mask_mod = EdgeTAMMaskDecoderExport(model).module.eval()

    image = cv2.cvtColor(cv2.imread(args.image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    input_image = preprocess_image(image, size=args.size).astype(np.float32)

    pts = _load_points_file(args.points_file, size=args.size)
    active = pts[: args.max_points]
    point_coords, point_labels = build_prompt_arrays(active, max_points=args.max_points, fixed_length=True)
    box_coords = np.zeros((1, 1, 4), dtype=np.float32)
    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)

    import torch

    ti = torch.from_numpy(input_image)
    tpc = torch.from_numpy(point_coords)
    tpl = torch.from_numpy(point_labels)
    tbc = torch.from_numpy(box_coords)
    tmi = torch.from_numpy(mask_input)

    with torch.no_grad():
        t_image_embeddings, t_high0, t_high1 = image_mod(ti)
        t_sparse, t_dense = prompt_mod(tpc, tpl, tbc, tmi)
        t_low, t_final, t_iou = mask_mod(t_image_embeddings, t_sparse, t_dense, t_high0, t_high1)

    providers = ["CPUExecutionProvider"]
    s_img = ort.InferenceSession(args.image_encoder_onnx, providers=providers)
    s_prompt = ort.InferenceSession(args.prompt_encoder_onnx, providers=providers)
    s_mask = ort.InferenceSession(args.mask_decoder_onnx, providers=providers)

    o_image_embeddings, o_high0, o_high1 = s_img.run(None, {"input_image": input_image})
    o_sparse, o_dense = s_prompt.run(
        None,
        {
            "point_coords": point_coords,
            "point_labels": point_labels,
            "box_coords": box_coords,
            "mask_input": mask_input,
        },
    )
    o_low, o_final, o_iou = s_mask.run(
        None,
        {
            "image_embeddings": o_image_embeddings,
            "sparse_prompt_embeddings": o_sparse,
            "dense_prompt_embeddings": o_dense,
            "high_res_feat_0": o_high0,
            "high_res_feat_1": o_high1,
        },
    )

    report = {
        "precision": args.precision,
        "rtol": rtol,
        "atol": atol,
        "inputs": {
            "image": args.image,
            "points_file": args.points_file,
            "max_points": args.max_points,
            "active_points": active,
        },
        "comparisons": {
            "image_embeddings": _compare(t_image_embeddings.cpu().numpy(), o_image_embeddings, rtol, atol),
            "high_res_feat_0": _compare(t_high0.cpu().numpy(), o_high0, rtol, atol),
            "high_res_feat_1": _compare(t_high1.cpu().numpy(), o_high1, rtol, atol),
            "prompt_sparse_embeddings": _compare(t_sparse.cpu().numpy(), o_sparse, rtol, atol),
            "prompt_dense_embeddings": _compare(t_dense.cpu().numpy(), o_dense, rtol, atol),
            "low_res_masks": _compare(t_low.cpu().numpy(), o_low, rtol, atol),
            "final_masks": _compare(t_final.cpu().numpy(), o_final, rtol, atol),
            "iou_scores": _compare(t_iou.cpu().numpy(), o_iou, rtol, atol),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    failures = [k for k, v in report["comparisons"].items() if not v["allclose"]]
    print(json.dumps({"report": str(out), "failed_tensors": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
