"""Benchmark split ONNX pipeline latency against PyTorch reference modules."""

from __future__ import annotations

import argparse
import json
import time
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


def _summary(samples_ms: list[float]) -> dict[str, float]:
    """Aggregate latency samples into human-readable metrics."""
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "std_ms": float(arr.std()),
        "fps_mean": float(1000.0 / arr.mean()) if arr.mean() > 0 else 0.0,
    }


def _bench(fn, warmup: int, runs: int) -> list[float]:
    """Benchmark callable latency in milliseconds after warmup iterations."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1_000_000.0)
    return samples


def main() -> int:
    """Run end-to-end and per-stage timing for split ONNX and PyTorch paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="point_mask_examples/reference_image.JPG")
    parser.add_argument("--points-file", default="point_mask_examples/bad6-points.txt")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--max-points", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--config", default="model/edgetam.yaml")
    parser.add_argument("--checkpoint", default="model/edgetam.pt")
    parser.add_argument("--sam2-root", default=None)
    parser.add_argument("--split-image-onnx", default="edgetam_onnx/models/image_encoder.onnx")
    parser.add_argument("--split-prompt-onnx", default="edgetam_onnx/models/prompt_encoder.onnx")
    parser.add_argument("--split-mask-onnx", default="edgetam_onnx/models/mask_decoder.onnx")
    parser.add_argument("--out", default="artifacts/benchmarks/split_vs_pytorch_bad6.json")
    args = parser.parse_args()

    sam2_root = resolve_sam2_root(args.sam2_root)
    setup_sam2_import_path(sam2_root)

    image = cv2.cvtColor(cv2.imread(args.image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    input_image = preprocess_image(image, size=args.size).astype(np.float32)

    pts = _load_points_file(args.points_file, size=args.size)
    active = pts[: args.max_points]
    point_coords, point_labels = build_prompt_arrays(active, max_points=args.max_points, fixed_length=True)
    box_coords = np.zeros((1, 1, 4), dtype=np.float32)
    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)

    providers = ["CPUExecutionProvider"]
    split_img = ort.InferenceSession(args.split_image_onnx, providers=providers)
    split_prompt = ort.InferenceSession(args.split_prompt_onnx, providers=providers)
    split_mask = ort.InferenceSession(args.split_mask_onnx, providers=providers)

    model = load_model(args.config, args.checkpoint)
    pyt_img = EdgeTAMImageEncoderExport(model).module.eval()
    pyt_prompt = EdgeTAMPromptEncoderExport(model).module.eval()
    pyt_mask = EdgeTAMMaskDecoderExport(model).module.eval()

    import torch

    ti = torch.from_numpy(input_image)
    tpc = torch.from_numpy(point_coords)
    tpl = torch.from_numpy(point_labels)
    tbc = torch.from_numpy(box_coords)
    tmi = torch.from_numpy(mask_input)

    def run_pytorch_full():
        with torch.no_grad():
            ie, h0, h1 = pyt_img(ti)
            sp, de = pyt_prompt(tpc, tpl, tbc, tmi)
            _ = pyt_mask(ie, sp, de, h0, h1)

    def run_split_full():
        ie, h0, h1 = split_img.run(None, {"input_image": input_image})
        sp, de = split_prompt.run(
            None,
            {
                "point_coords": point_coords,
                "point_labels": point_labels,
                "box_coords": box_coords,
                "mask_input": mask_input,
            },
        )
        _ = split_mask.run(
            None,
            {
                "image_embeddings": ie,
                "sparse_prompt_embeddings": sp,
                "dense_prompt_embeddings": de,
                "high_res_feat_0": h0,
                "high_res_feat_1": h1,
            },
        )

    # Stage timings for split.
    def run_split_img():
        _ = split_img.run(None, {"input_image": input_image})

    cached_ie, cached_h0, cached_h1 = split_img.run(None, {"input_image": input_image})

    def run_split_prompt():
        _ = split_prompt.run(
            None,
            {
                "point_coords": point_coords,
                "point_labels": point_labels,
                "box_coords": box_coords,
                "mask_input": mask_input,
            },
        )

    cached_sp, cached_de = split_prompt.run(
        None,
        {
            "point_coords": point_coords,
            "point_labels": point_labels,
            "box_coords": box_coords,
            "mask_input": mask_input,
        },
    )

    def run_split_mask():
        _ = split_mask.run(
            None,
            {
                "image_embeddings": cached_ie,
                "sparse_prompt_embeddings": cached_sp,
                "dense_prompt_embeddings": cached_de,
                "high_res_feat_0": cached_h0,
                "high_res_feat_1": cached_h1,
            },
        )

    pytorch_samples = _bench(run_pytorch_full, args.warmup, args.runs)
    split_samples = _bench(run_split_full, args.warmup, args.runs)

    split_img_samples = _bench(run_split_img, args.warmup, args.runs)
    split_prompt_samples = _bench(run_split_prompt, args.warmup, args.runs)
    split_mask_samples = _bench(run_split_mask, args.warmup, args.runs)

    report = {
        "meta": {
            "image": args.image,
            "points_file": args.points_file,
            "size": args.size,
            "max_points": args.max_points,
            "warmup": args.warmup,
            "runs": args.runs,
            "providers": providers,
            "active_points": active,
        },
        "summary": {
            "pytorch_split_reference": _summary(pytorch_samples),
            "new_split_onnx_pipeline": _summary(split_samples),
            "new_split_onnx_stages": {
                "image_encoder": _summary(split_img_samples),
                "prompt_encoder": _summary(split_prompt_samples),
                "mask_decoder": _summary(split_mask_samples),
            },
        },
        "derived": {
            "split_vs_pytorch_speedup": float(np.mean(pytorch_samples) / np.mean(split_samples)),
            "pytorch_vs_split_speed_ratio": float(np.mean(split_samples) / np.mean(pytorch_samples)),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
