from __future__ import annotations

import argparse
from pathlib import Path

import torch

from edgetam_onnx.export.common import EdgeTAMImageEncoderExport
from edgetam_onnx.export.common import ensure_export_deps
from edgetam_onnx.export.common import load_model
from edgetam_onnx.export.common import resolve_sam2_root
from edgetam_onnx.export.common import setup_sam2_import_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="model/edgetam.yaml")
    parser.add_argument("--checkpoint", default="model/edgetam.pt")
    parser.add_argument("--output", default="edgetam_onnx/models/image_encoder.onnx")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--sam2-root", default=None)
    parser.add_argument("--allow-backbone-download", action="store_true")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    sam2_root = resolve_sam2_root(args.sam2_root)
    setup_sam2_import_path(sam2_root)
    ensure_export_deps()

    model = load_model(args.config, args.checkpoint, allow_backbone_download=args.allow_backbone_download)
    wrapper = EdgeTAMImageEncoderExport(model).module.eval()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(1, 3, args.size, args.size, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out),
        input_names=["input_image"],
        output_names=["image_embeddings", "high_res_feat_0", "high_res_feat_1"],
        opset_version=args.opset,
        dynamo=True,
        external_data=False,
    )
    print(f"Exported image encoder to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
