from __future__ import annotations

import argparse
from pathlib import Path

import torch

from edgetam_onnx.export.common import EdgeTAMMaskDecoderExport
from edgetam_onnx.export.common import ensure_export_deps
from edgetam_onnx.export.common import load_model
from edgetam_onnx.export.common import resolve_sam2_root
from edgetam_onnx.export.common import setup_sam2_import_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="model/edgetam.yaml")
    parser.add_argument("--checkpoint", default="model/edgetam.pt")
    parser.add_argument("--output", default="edgetam_onnx/models/mask_decoder.onnx")
    parser.add_argument("--max-points", type=int, default=4)
    parser.add_argument("--sam2-root", default=None)
    parser.add_argument("--allow-backbone-download", action="store_true")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    sam2_root = resolve_sam2_root(args.sam2_root)
    setup_sam2_import_path(sam2_root)
    ensure_export_deps()

    model = load_model(args.config, args.checkpoint, allow_backbone_download=args.allow_backbone_download)
    wrapper = EdgeTAMMaskDecoderExport(model).module.eval()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    sparse_tokens = args.max_points + 1
    image_embeddings = torch.randn(1, 256, 64, 64, dtype=torch.float32)
    sparse_prompt_embeddings = torch.randn(1, sparse_tokens, 256, dtype=torch.float32)
    dense_prompt_embeddings = torch.randn(1, 256, 64, 64, dtype=torch.float32)
    high_res_feat_0 = torch.randn(1, 32, 256, 256, dtype=torch.float32)
    high_res_feat_1 = torch.randn(1, 64, 128, 128, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (
            image_embeddings,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
            high_res_feat_0,
            high_res_feat_1,
        ),
        str(out),
        input_names=[
            "image_embeddings",
            "sparse_prompt_embeddings",
            "dense_prompt_embeddings",
            "high_res_feat_0",
            "high_res_feat_1",
        ],
        output_names=["low_res_masks", "final_masks", "iou_scores"],
        opset_version=args.opset,
        dynamo=True,
        external_data=False,
    )
    print(f"Exported mask decoder to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
