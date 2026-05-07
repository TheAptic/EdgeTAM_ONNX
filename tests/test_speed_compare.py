from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from edgetam_onnx.runner import EdgeTamOnnxRunner, preprocess_image
from scripts.prompt_inputs import build_prompt_arrays


def _load_image_rgb(image_path: str, size: int) -> np.ndarray:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR)


def _load_edge_mask(mask_path: str, size: int) -> np.ndarray:
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Failed to read edge mask: {mask_path}")
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _summary_ms(samples_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    mean = float(arr.mean())
    return {
        "runs": float(arr.size),
        "mean_ms": mean,
        "median_ms": float(np.median(arr)),
        "p90_ms": _percentile(samples_ms, 90),
        "p95_ms": _percentile(samples_ms, 95),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "std_ms": float(arr.std()),
        "fps_mean": 1000.0 / mean if mean > 0.0 else 0.0,
        "images_per_sec_mean": (1000.0 / mean) if mean > 0.0 else 0.0,
    }


def _summary_per_image(samples_ms: list[float], batch_size: int) -> dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64) / max(batch_size, 1)
    mean = float(arr.mean())
    return {
        "mean_ms_per_image": mean,
        "median_ms_per_image": float(np.median(arr)),
        "p95_ms_per_image": float(np.percentile(arr, 95)),
        "images_per_sec_total": (1000.0 / mean) if mean > 0.0 else 0.0,
    }


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = (a > 0).astype(np.uint8)
    bb = (b > 0).astype(np.uint8)
    inter = int((aa & bb).sum())
    union = int((aa | bb).sum())
    if union == 0:
        return 1.0
    return inter / union


def _point_in_mask(mask: np.ndarray, x: int, y: int) -> bool:
    h, w = mask.shape
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return bool(mask[y, x])


def _select_mask_polarity(logits_2d: np.ndarray, points: list[tuple[int, int, int]]) -> tuple[np.ndarray, str]:
    mask_neg = (logits_2d < 0).astype(np.uint8)
    mask_pos = (logits_2d > 0).astype(np.uint8)

    def _score(mask: np.ndarray) -> int:
        score = 0
        for x, y, label in points:
            inside = _point_in_mask(mask, x, y)
            if label > 0:
                score += 1 if inside else -1
            else:
                score += 1 if not inside else -1
        return score

    score_neg = _score(mask_neg)
    score_pos = _score(mask_pos)
    if score_neg > score_pos:
        return mask_neg, "lt0"
    if score_pos > score_neg:
        return mask_pos, "gt0"
    if int(mask_pos.sum()) < int(mask_neg.sum()):
        return mask_pos, "gt0_tiebreak"
    return mask_neg, "lt0_tiebreak"


def _percent_points_to_pixels(size: int) -> list[tuple[int, int, int]]:
    positives = [
        (39.46, 79.48),
        (39.13, 45.04),
        (56.03, 17.34),
        (57.75, 84.46),
        (62.80, 34.64),
        (49.40, 91.42),
    ]
    negatives = [
        (5.86, 80.01),
        (19.40, 24.06),
        (32.93, 18.72),
        (25.14, 72.81),
        (76.03, 17.57),
        (74.96, 52.87),
        (73.90, 80.89),
        (86.34, 85.43),
        (87.78, 20.27),
    ]
    out: list[tuple[int, int, int]] = []
    for x_pct, y_pct in positives:
        x = int(np.clip((x_pct / 100.0) * size, 0, size - 1))
        y = int(np.clip((y_pct / 100.0) * size, 0, size - 1))
        out.append((x, y, 1))
    for x_pct, y_pct in negatives:
        x = int(np.clip((x_pct / 100.0) * size, 0, size - 1))
        y = int(np.clip((y_pct / 100.0) * size, 0, size - 1))
        out.append((x, y, 0))
    return out


def _build_prompt_scenarios(size: int) -> dict[str, list[tuple[int, int, int]]]:
    app_points = _percent_points_to_pixels(size)
    c = size // 2
    return {
        "single_positive": [(c, c, 1)],
        "two_pos_one_neg": [(c, c, 1), (c + 90, c - 20, 1), (c + 120, c + 40, 0)],
        "eight_mixed": app_points[:8],
        "app_full_15pts": app_points,
    }


def _parse_batch_sizes(raw: str) -> list[int]:
    vals = []
    for token in raw.split(","):
        t = token.strip()
        if not t:
            continue
        vals.append(max(1, int(t)))
    if not vals:
        raise ValueError("No batch sizes provided")
    return vals


def _build_coreml_provider_options(args: argparse.Namespace) -> dict[str, str]:
    opts = {
        "ModelFormat": args.coreml_model_format,
        "MLComputeUnits": args.coreml_compute_units,
        "RequireStaticInputShapes": str(int(args.coreml_require_static_input_shapes)),
        "EnableOnSubgraphs": str(int(args.coreml_enable_on_subgraphs)),
    }
    if args.coreml_cache_dir:
        opts["ModelCacheDirectory"] = args.coreml_cache_dir
    return opts


def _select_onnx_providers(backend_mode: str, args: argparse.Namespace) -> tuple[list[Any], str]:
    available = set(ort.get_available_providers())
    if backend_mode == "cpu":
        return ["CPUExecutionProvider"], "CPUExecutionProvider"
    # best: prefer CoreML EP on macOS if available, fallback CPU
    if "CoreMLExecutionProvider" in available:
        coreml_opts = _build_coreml_provider_options(args)
        return [("CoreMLExecutionProvider", coreml_opts), "CPUExecutionProvider"], "CoreMLExecutionProvider"
    return ["CPUExecutionProvider"], "CPUExecutionProvider"


def _build_onnx_runner_with_fallback(
    model_path: str,
    backend_mode: str,
    args: argparse.Namespace,
) -> tuple[EdgeTamOnnxRunner, dict[str, Any]]:
    providers, preferred = _select_onnx_providers(backend_mode, args=args)
    runner = EdgeTamOnnxRunner(model_path, providers=providers)
    meta: dict[str, Any] = {
        "preferred_provider": preferred,
        "requested_providers": providers,
        "actual_providers": runner.session.get_providers(),
        "fallback_used": False,
        "fallback_reason": "",
    }
    if backend_mode != "best" or preferred != "CoreMLExecutionProvider":
        return runner, meta
    try:
        # Minimal probe to detect EP runtime failures early.
        probe_img = np.zeros((1, 3, 1024, 1024), dtype=np.float32)
        probe_coords = np.zeros((1, 20, 2), dtype=np.float32)
        probe_labels = np.full((1, 20), -1.0, dtype=np.float32)
        probe_coords[0, 0] = [512.0, 512.0]
        probe_labels[0, 0] = 1.0
        _ = runner.session.run(
            None,
            {"image": probe_img, "point_coords": probe_coords, "point_labels": probe_labels},
        )
        return runner, meta
    except Exception as exc:
        fallback_runner = EdgeTamOnnxRunner(model_path, providers=["CPUExecutionProvider"])
        meta["fallback_used"] = True
        meta["fallback_reason"] = str(exc).splitlines()[0]
        meta["actual_providers_after_fallback"] = fallback_runner.session.get_providers()
        return fallback_runner, meta


def _load_torch_model(config: str, checkpoint: str):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config)
    model = instantiate(cfg.model, _recursive_=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def _to_torch_device(model, preferred_backend: str):
    import torch

    if preferred_backend == "best" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return model.to(device), device


def _torch_run_once(model, device, image_nchw: np.ndarray, coords: np.ndarray, labels: np.ndarray):
    import torch

    image = torch.from_numpy(image_nchw).to(device=device, dtype=torch.float32)
    point_coords = torch.from_numpy(coords).to(device=device, dtype=torch.float32)
    point_labels = torch.from_numpy(labels).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        backbone_out = model.forward_image(image)
        _, vision_feats, vision_pos_embeds, feat_sizes = model._prepare_backbone_features(backbone_out)
        out = model.track_step(
            frame_idx=0,
            is_init_cond_frame=True,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs={"point_coords": point_coords, "point_labels": point_labels},
            mask_inputs=None,
            output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            num_frames=image.shape[0],
            run_mem_encoder=False,
        )
        logits = out["pred_masks_high_res"].detach().to("cpu").numpy()
    return logits


def _onnx_run_once(runner: EdgeTamOnnxRunner, image_nchw: np.ndarray, coords: np.ndarray, labels: np.ndarray):
    outputs = runner.session.run(
        None,
        {
            "image": image_nchw.astype(np.float32, copy=False),
            "point_coords": coords.astype(np.float32, copy=False),
            "point_labels": labels.astype(np.float32, copy=False),
        },
    )
    out_names = [o.name for o in runner.session.get_outputs()]
    out = dict(zip(out_names, outputs, strict=True))
    return out["pred_masks_high_res"]


def _bench(fn, warmup_runs: int, timed_runs: int) -> list[float]:
    for _ in range(warmup_runs):
        fn()
    samples = []
    for _ in range(timed_runs):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1_000_000.0)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Best-backend ONNX vs PyTorch throughput benchmark on macOS."
    )
    parser.add_argument("--image", default="DSC00552.JPG")
    parser.add_argument("--edge-mask", default="resources/edge-mask.png")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--onnx", default="model/inference_bundle/edgetam_prompt20.onnx")
    parser.add_argument("--config", default="model/edgetam.yaml")
    parser.add_argument("--checkpoint", default="model/edgetam.pt")
    parser.add_argument("--max-points", type=int, default=20)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--scenario", default="app_full_15pts")
    parser.add_argument("--warmup-runs", type=int, default=20)
    parser.add_argument("--timed-runs", type=int, default=100)
    parser.add_argument("--backend", choices=["best", "cpu"], default="best")
    parser.add_argument(
        "--coreml-model-format",
        choices=["NeuralNetwork", "MLProgram"],
        default="NeuralNetwork",
    )
    parser.add_argument(
        "--coreml-compute-units",
        choices=["ALL", "CPUAndGPU", "CPUOnly", "CPUAndNeuralEngine"],
        default="ALL",
    )
    parser.add_argument("--coreml-require-static-input-shapes", type=int, choices=[0, 1], default=1)
    parser.add_argument("--coreml-enable-on-subgraphs", type=int, choices=[0, 1], default=0)
    parser.add_argument("--coreml-cache-dir", default="artifacts/coreml_cache")
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--sam2-root", default="EdgeTAM")
    parser.add_argument("--out", default="artifacts/benchmarks/pytorch_vs_onnx_best_backend.json")
    args = parser.parse_args()

    sam2_root = Path(args.sam2_root)
    if sam2_root.exists():
        sys.path.insert(0, str(sam2_root))

    image_rgb = _load_image_rgb(args.image, args.size)
    gt_mask = _load_edge_mask(args.edge_mask, args.size)
    scenarios = _build_prompt_scenarios(args.size)
    if args.scenario not in scenarios:
        raise ValueError(f"Unknown scenario {args.scenario}. Options: {sorted(scenarios.keys())}")
    points = scenarios[args.scenario]
    coords_1, labels_1 = build_prompt_arrays(points, max_points=args.max_points, fixed_length=True)

    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    onnx_runner, onnx_meta = _build_onnx_runner_with_fallback(args.onnx, args.backend, args)

    import torch
    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
        torch.set_num_interop_threads(max(1, min(args.torch_num_threads, 8)))
    torch_model = _load_torch_model(args.config, args.checkpoint)
    torch_model, torch_device = _to_torch_device(torch_model, args.backend)

    # shared preprocessing once for throughput-kernel style measurement
    image_1 = preprocess_image(image_rgb, size=args.size)

    results: list[dict[str, Any]] = []
    quality: dict[str, Any] = {}
    onnx_runtime_failures: list[dict[str, Any]] = []
    for b in batch_sizes:
        image_b = np.repeat(image_1, b, axis=0)
        coords_b = np.repeat(coords_1, b, axis=0)
        labels_b = np.repeat(labels_1, b, axis=0)

        try:
            onnx_ms = _bench(
                lambda: _onnx_run_once(onnx_runner, image_b, coords_b, labels_b),
                warmup_runs=args.warmup_runs,
                timed_runs=args.timed_runs,
            )
        except Exception as exc:
            if args.backend == "best":
                onnx_runtime_failures.append(
                    {
                        "batch_size": b,
                        "error": str(exc).splitlines()[0],
                        "providers_before_fallback": onnx_runner.session.get_providers(),
                    }
                )
                onnx_runner = EdgeTamOnnxRunner(args.onnx, providers=["CPUExecutionProvider"])
                onnx_meta["fallback_used"] = True
                if not onnx_meta.get("fallback_reason"):
                    onnx_meta["fallback_reason"] = str(exc).splitlines()[0]
                onnx_meta["actual_providers_after_fallback"] = onnx_runner.session.get_providers()
                onnx_ms = _bench(
                    lambda: _onnx_run_once(onnx_runner, image_b, coords_b, labels_b),
                    warmup_runs=args.warmup_runs,
                    timed_runs=args.timed_runs,
                )
            else:
                raise
        torch_ms = _bench(
            lambda: _torch_run_once(torch_model, torch_device, image_b, coords_b, labels_b),
            warmup_runs=args.warmup_runs,
            timed_runs=args.timed_runs,
        )

        onnx_stats = _summary_ms(onnx_ms)
        onnx_stats.update(_summary_per_image(onnx_ms, b))
        torch_stats = _summary_ms(torch_ms)
        torch_stats.update(_summary_per_image(torch_ms, b))

        row = {
            "batch_size": b,
            "onnx": onnx_stats,
            "pytorch": torch_stats,
            "speedup_onnx_vs_pytorch_by_mean_ms_per_image": (
                torch_stats["mean_ms_per_image"] / onnx_stats["mean_ms_per_image"]
                if onnx_stats["mean_ms_per_image"] > 0.0
                else 0.0
            ),
            "throughput_ratio_onnx_vs_pytorch": (
                onnx_stats["images_per_sec_total"] / torch_stats["images_per_sec_total"]
                if torch_stats["images_per_sec_total"] > 0.0
                else 0.0
            ),
        }
        results.append(row)

        if b == 1:
            onnx_logits = _onnx_run_once(onnx_runner, image_b, coords_b, labels_b)[0, 0]
            torch_logits = _torch_run_once(torch_model, torch_device, image_b, coords_b, labels_b)[0, 0]
            onnx_mask, onnx_polarity = _select_mask_polarity(onnx_logits, points)
            torch_mask, torch_polarity = _select_mask_polarity(torch_logits, points)
            quality = {
                "scenario": args.scenario,
                "point_count": len(points),
                "onnx_polarity": onnx_polarity,
                "torch_polarity": torch_polarity,
                "iou_onnx_vs_pytorch": _iou(onnx_mask, torch_mask),
                "iou_onnx_vs_edge_mask": _iou(onnx_mask, gt_mask),
                "iou_torch_vs_edge_mask": _iou(torch_mask, gt_mask),
            }

    report = {
        "meta": {
            "host": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "image": args.image,
            "edge_mask": args.edge_mask,
            "size": args.size,
            "onnx_model": args.onnx,
            "config": args.config,
            "checkpoint": args.checkpoint,
            "max_points": args.max_points,
            "scenario": args.scenario,
            "points_xy_label": points,
            "warmup_runs": args.warmup_runs,
            "timed_runs": args.timed_runs,
            "batch_sizes": batch_sizes,
            "backend_mode": args.backend,
            "onnx_provider_meta": onnx_meta,
            "coreml_provider_options": _build_coreml_provider_options(args),
            "onnx_runtime_failures": onnx_runtime_failures,
            "torch_device": str(torch_device),
            "torch_num_threads": int(torch.get_num_threads()),
        },
        "quality_batch1": quality,
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Benchmark complete: {out_path}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
