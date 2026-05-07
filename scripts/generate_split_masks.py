from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from edgetam_onnx.runner import preprocess_image
from scripts.onnx_parity_harness import _foreground_mask_from_logits
from scripts.onnx_parity_harness import _load_points_file
from scripts.prompt_inputs import build_prompt_arrays


def _unique_out_path(base: Path) -> Path:
    if not base.exists():
        return base
    stem = base.stem
    suffix = base.suffix
    parent = base.parent
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="point_mask_examples/reference_image.JPG")
    parser.add_argument("--points-dir", default="point_mask_examples")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--max-points", type=int, default=4)
    parser.add_argument("--image-encoder", default="edgetam_onnx/models/image_encoder.onnx")
    parser.add_argument("--prompt-encoder", default="edgetam_onnx/models/prompt_encoder.onnx")
    parser.add_argument("--mask-decoder", default="edgetam_onnx/models/mask_decoder.onnx")
    args = parser.parse_args()

    points_dir = Path(args.points_dir)
    txt_files = sorted(points_dir.glob("*.txt"))
    if not txt_files:
        raise SystemExit(f"No .txt points files found in {points_dir}")

    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(args.image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    input_image = preprocess_image(image_rgb, size=args.size).astype(np.float32)

    providers = ["CPUExecutionProvider"]
    s_img = ort.InferenceSession(args.image_encoder, providers=providers)
    s_prompt = ort.InferenceSession(args.prompt_encoder, providers=providers)
    s_mask = ort.InferenceSession(args.mask_decoder, providers=providers)

    image_embeddings, high_res_feat_0, high_res_feat_1 = s_img.run(None, {"input_image": input_image})

    wrote = []
    for txt in txt_files:
        points = _load_points_file(str(txt), size=args.size)
        active_points = points[: args.max_points]
        point_coords, point_labels = build_prompt_arrays(
            active_points,
            max_points=args.max_points,
            fixed_length=True,
        )
        box_coords = np.zeros((1, 1, 4), dtype=np.float32)
        mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)

        sparse_embeddings, dense_embeddings = s_prompt.run(
            None,
            {
                "point_coords": point_coords.astype(np.float32),
                "point_labels": point_labels.astype(np.float32),
                "box_coords": box_coords,
                "mask_input": mask_input,
            },
        )

        low_res_masks, final_masks, _iou_scores = s_mask.run(
            None,
            {
                "image_embeddings": image_embeddings,
                "sparse_prompt_embeddings": sparse_embeddings,
                "dense_prompt_embeddings": dense_embeddings,
                "high_res_feat_0": high_res_feat_0,
                "high_res_feat_1": high_res_feat_1,
            },
        )

        logits = final_masks[0, 0]
        mask_bin = _foreground_mask_from_logits(logits, active_points=active_points).astype(np.uint8) * 255

        base_name = txt.stem
        out_path = _unique_out_path(points_dir / f"{base_name}_split_new.png")
        cv2.imwrite(str(out_path), mask_bin)
        wrote.append({
            "points_file": str(txt),
            "out_mask": str(out_path),
            "used_points": len(active_points),
            "total_points": len(points),
        })

    for row in wrote:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
