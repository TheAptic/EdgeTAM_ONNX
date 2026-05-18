"""CLI to export prompt encoding stage for split ONNX runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from edgetam_onnx.export.common import EdgeTAMPromptEncoderExport
from edgetam_onnx.export.common import ensure_export_deps
from edgetam_onnx.export.common import load_model
from edgetam_onnx.export.common import resolve_sam2_root
from edgetam_onnx.export.common import setup_sam2_import_path


def main() -> int:
    """Export prompt encoder with static prompt slots for deployment stability."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="model/edgetam.yaml")
    parser.add_argument("--checkpoint", default="model/edgetam.pt")
    parser.add_argument("--output", default="edgetam_onnx/models/prompt_encoder.onnx")
    parser.add_argument("--max-points", type=int, default=4)
    parser.add_argument("--sam2-root", default=None)
    parser.add_argument("--allow-backbone-download", action="store_true")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    sam2_root = resolve_sam2_root(args.sam2_root)
    setup_sam2_import_path(sam2_root)
    ensure_export_deps()

    model = load_model(args.config, args.checkpoint, allow_backbone_download=args.allow_backbone_download)
    wrapper = EdgeTAMPromptEncoderExport(model).module.eval()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    point_coords = torch.zeros(1, args.max_points, 2, dtype=torch.float32)
    point_coords[0, 0] = torch.tensor([512.0, 512.0])
    point_labels = torch.full((1, args.max_points), -1.0, dtype=torch.float32)
    point_labels[0, 0] = 1.0
    box_coords = torch.zeros(1, 1, 4, dtype=torch.float32)
    mask_input = torch.zeros(1, 1, 256, 256, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (point_coords, point_labels, box_coords, mask_input),
        str(out),
        input_names=["point_coords", "point_labels", "box_coords", "mask_input"],
        output_names=["sparse_embeddings", "dense_embeddings"],
        opset_version=args.opset,
        dynamo=True,
        external_data=False,
    )
    print(f"Exported prompt encoder to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
