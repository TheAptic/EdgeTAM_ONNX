"""Export a monolithic EdgeTAM ONNX graph from SAM2 config + checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import importlib
from pathlib import Path
import sys
from typing import Any


def _is_filesystem_config(config: str) -> bool:
    """Return True when `config` is an existing local YAML path."""
    p = Path(config)
    return p.suffix in {".yaml", ".yml"} and p.exists()


def _collect_targets(node: Any, out: list[str]) -> None:
    """Collect Hydra `_target_` strings recursively for import preflight checks."""
    if isinstance(node, dict):
        target = node.get("_target_")
        if isinstance(target, str):
            out.append(target)
        for v in node.values():
            _collect_targets(v, out)
    elif isinstance(node, list):
        for item in node:
            _collect_targets(item, out)


def _resolve_target(target: str) -> bool:
    """Best-effort resolve a dotted object path without instantiating it."""
    parts = target.split(".")
    for i in range(len(parts), 0, -1):
        module_name = ".".join(parts[:i])
        try:
            module = importlib.import_module(module_name)
            obj = module
            for part in parts[i:]:
                obj = getattr(obj, part)
            return True
        except Exception:
            continue
    return False


def _sanitize_point_labels_for_onnx(point_labels, torch_module):
    """Map labels to ONNX contract: `1` foreground, `0` background, `-1` empty."""
    ones = torch_module.ones_like(point_labels)
    zeros = torch_module.zeros_like(point_labels)
    neg_one = torch_module.full_like(point_labels, -1)
    return torch_module.where(
        point_labels == -1,
        neg_one,
        torch_module.where(point_labels > 0, ones, zeros),
    )


def main() -> int:
    """Export ONNX and validate that the local SAM2 checkout matches config targets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="model/edgetam.yaml")
    parser.add_argument("--checkpoint", default="model/edgetam.pt")
    parser.add_argument("--output", default="model/edgetam.onnx")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument(
        "--max-points",
        type=int,
        default=4,
        help="Fixed prompt slots exported in ONNX inputs.",
    )
    parser.add_argument(
        "--dynamic-points",
        action="store_true",
        help="Export with dynamic prompt length axis for point inputs.",
    )
    parser.add_argument(
        "--force-single-mask",
        action="store_true",
        help="Disable SAM multimask branch during export path for stable prompt semantics.",
    )
    parser.add_argument(
        "--sam2-root",
        default=None,
        help="Path to SAM2/EdgeTAM repo root that contains the `sam2` package",
    )
    parser.add_argument(
        "--allow-backbone-download",
        action="store_true",
        help="Allow timm backbone constructors to download pretrained weights.",
    )
    parser.add_argument(
        "--exporter",
        choices=["legacy", "dynamo"],
        default="legacy",
        help="ONNX exporter path: legacy uses dynamo=False, dynamo uses dynamo=True with optional diagnostics.",
    )
    parser.add_argument(
        "--dynamo-report",
        action="store_true",
        help="Enable ONNX dynamo export report generation (only with --exporter dynamo).",
    )
    parser.add_argument(
        "--dynamo-verify",
        action="store_true",
        help="Enable ONNX dynamo export verification (only with --exporter dynamo).",
    )
    parser.add_argument(
        "--dynamo-profile",
        action="store_true",
        help="Enable ONNX dynamo export profiling (only with --exporter dynamo).",
    )
    parser.add_argument(
        "--dynamo-artifacts-dir",
        default=None,
        help="Directory for ONNX dynamo export artifacts (only with --exporter dynamo).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    default_root = project_root / "EdgeTAM" if (project_root / "EdgeTAM").exists() else project_root / "sam2"
    sam2_root = Path(args.sam2_root) if args.sam2_root else default_root
    if not sam2_root.exists():
        raise SystemExit(f"--sam2-root does not exist: {sam2_root}")
    sys.path.insert(0, str(sam2_root))

    missing = []
    if importlib.util.find_spec("torch") is None:
        missing.append("torch")
    if importlib.util.find_spec("sam2") is None:
        missing.append("sam2")

    if missing:
        missing_list = ", ".join(missing)
        raise SystemExit(
            "Missing export dependency/dependencies: "
            f"{missing_list}\n\n"
            "Install into this venv, then rerun:\n"
            "  .venv/bin/python -m pip install torch\n"
            "  # install SAM2 from a local checkout:\n"
            "  .venv/bin/python -m pip install -e /path/to/segment-anything-2\n\n"
            "Export requires sam2 + torch only for conversion. "
            "Runtime inference remains sam2-free."
        )

    import torch
    from hydra import compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    def _patch_prompt_encoder_for_onnx() -> None:
        try:
            prompt_mod = importlib.import_module("sam2.modeling.sam.prompt_encoder")
            PromptEncoder = prompt_mod.PromptEncoder
            if getattr(PromptEncoder, "_edgetam_onnx_patch_applied", False):
                return

            def _embed_points_onnx_safe(self, points, labels, pad):
                points = points + 0.5
                if pad:
                    padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
                    padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
                    points = torch.cat([points, padding_point], dim=1)
                    labels = torch.cat([labels, padding_label], dim=1)

                point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
                labels = labels.to(point_embedding.dtype)
                is_neg1 = (labels == -1).to(point_embedding.dtype).unsqueeze(-1)
                is_0 = (labels == 0).to(point_embedding.dtype).unsqueeze(-1)
                is_1 = (labels == 1).to(point_embedding.dtype).unsqueeze(-1)
                is_2 = (labels == 2).to(point_embedding.dtype).unsqueeze(-1)
                is_3 = (labels == 3).to(point_embedding.dtype).unsqueeze(-1)

                point_embedding = point_embedding * (1.0 - is_neg1)
                point_embedding = point_embedding + is_neg1 * self.not_a_point_embed.weight.reshape(1, 1, -1)
                point_embedding = point_embedding + is_0 * self.point_embeddings[0].weight.reshape(1, 1, -1)
                point_embedding = point_embedding + is_1 * self.point_embeddings[1].weight.reshape(1, 1, -1)
                point_embedding = point_embedding + is_2 * self.point_embeddings[2].weight.reshape(1, 1, -1)
                point_embedding = point_embedding + is_3 * self.point_embeddings[3].weight.reshape(1, 1, -1)
                return point_embedding

            def _forward_onnx_safe(self, points, boxes, masks):
                bs = self._get_batch_size(points, boxes, masks)
                pieces = []
                if points is not None:
                    coords, labels = points
                    pieces.append(self._embed_points(coords, labels, pad=(boxes is None)))
                if boxes is not None:
                    pieces.append(self._embed_boxes(boxes))
                if pieces:
                    sparse_embeddings = torch.cat(pieces, dim=1) if len(pieces) > 1 else pieces[0]
                else:
                    # Keep API behavior for prompt-free calls while avoiding 0-length cat chains.
                    sparse_embeddings = torch.zeros(
                        (bs, 1, self.embed_dim),
                        device=self._get_device(),
                        dtype=self.point_embeddings[0].weight.dtype,
                    )
                    sparse_embeddings = sparse_embeddings[:, :0, :]

                if masks is not None:
                    dense_embeddings = self._embed_masks(masks)
                else:
                    dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                        bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
                    )
                return sparse_embeddings, dense_embeddings

            PromptEncoder._embed_points = _embed_points_onnx_safe
            PromptEncoder.forward = _forward_onnx_safe
            PromptEncoder._edgetam_onnx_patch_applied = True
        except Exception:
            pass

    class _SingleFrameSam2Export(torch.nn.Module):
        def __init__(self, sam_model, max_points: int, dynamic_points: bool):
            super().__init__()
            self.sam_model = sam_model
            self.max_points = max_points
            self.dynamic_points = dynamic_points

        def _sanitize_point_labels(self, point_labels: torch.Tensor) -> torch.Tensor:
            # ONNX contract: 1=positive, 0=negative, -1=unused slot.
            # Accept app-side enums by mapping any value >0 to 1.
            return _sanitize_point_labels_for_onnx(point_labels, torch_module=torch)

        def forward(self, image, point_coords, point_labels):
            backbone_out = self.sam_model.forward_image(image)
            _, vision_feats, vision_pos_embeds, feat_sizes = (
                self.sam_model._prepare_backbone_features(backbone_out)
            )
            if not self.dynamic_points:
                if point_coords.shape[1] != self.max_points:
                    raise ValueError("point_coords shape[1] must match exported max_points")
                if point_labels.shape[1] != self.max_points:
                    raise ValueError("point_labels shape[1] must match exported max_points")
            point_labels = self._sanitize_point_labels(point_labels)
            out = self.sam_model.track_step(
                frame_idx=0,
                is_init_cond_frame=True,
                current_vision_feats=vision_feats,
                current_vision_pos_embeds=vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs={
                    "point_coords": point_coords,
                    "point_labels": point_labels,
                },
                mask_inputs=None,
                output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
                num_frames=1,
                run_mem_encoder=False,
            )
            return out["pred_masks_high_res"], out["pred_masks"], out["object_score_logits"]

    if _is_filesystem_config(args.config):
        cfg = OmegaConf.load(args.config)
    else:
        cfg = compose(config_name=args.config)
    OmegaConf.resolve(cfg)

    if not args.allow_backbone_download:
        try:
            timm_backbone_mod = importlib.import_module("sam2.modeling.backbones.timm")
            original_create_model = timm_backbone_mod.create_model

            def _offline_create_model(*model_args, **model_kwargs):
                model_kwargs["pretrained"] = False
                return original_create_model(*model_args, **model_kwargs)

            timm_backbone_mod.create_model = _offline_create_model
        except Exception:
            pass

    _patch_prompt_encoder_for_onnx()

    targets: list[str] = []
    _collect_targets(OmegaConf.to_container(cfg, resolve=True), targets)
    missing_targets = sorted({t for t in targets if not _resolve_target(t)})
    if missing_targets:
        formatted = "\n".join(f"  - {t}" for t in missing_targets[:30])
        raise SystemExit(
            "Your current SAM2 codebase is missing model classes required by this config.\n"
            "Missing _target_ entries include:\n"
            f"{formatted}\n\n"
            "Install/use the exact Edgetam-compatible SAM2 fork used for training this checkpoint, "
            "then rerun export."
        )

    model = instantiate(cfg.model, _recursive_=True)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
    if missing_keys:
        raise RuntimeError(f"Missing checkpoint keys: {missing_keys[:10]}")
    if unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected_keys[:10]}")
    model.eval()
    if args.force_single_mask:
        if hasattr(model, "multimask_output_in_sam"):
            model.multimask_output_in_sam = False
        if hasattr(model, "multimask_output_for_tracking"):
            model.multimask_output_for_tracking = False
    export_model = _SingleFrameSam2Export(
        model,
        max_points=args.max_points,
        dynamic_points=args.dynamic_points,
    ).eval()

    dummy = torch.zeros(1, 3, args.size, args.size)
    dummy_n_points = 1 if args.dynamic_points else args.max_points
    dummy_coords = torch.zeros(1, dummy_n_points, 2, dtype=torch.float32)
    dummy_labels = torch.full((1, dummy_n_points), -1.0, dtype=torch.float32)
    dummy_coords[:, 0, 0] = args.size / 2.0
    dummy_coords[:, 0, 1] = args.size / 2.0
    dummy_labels[:, 0] = 1.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_axes = {
        "image": {0: "batch"},
        "point_coords": {0: "batch"},
        "point_labels": {0: "batch"},
        "pred_masks_high_res": {0: "batch"},
        "pred_masks_low_res": {0: "batch"},
        "object_score_logits": {0: "batch"},
    }
    if args.dynamic_points:
        dynamic_axes["point_coords"][1] = "num_points"
        dynamic_axes["point_labels"][1] = "num_points"

    export_kwargs = {
        "input_names": ["image", "point_coords", "point_labels"],
        "output_names": ["pred_masks_high_res", "pred_masks_low_res", "object_score_logits"],
        "opset_version": 17,
        "external_data": False,
    }
    if args.exporter == "dynamo":
        export_kwargs["opset_version"] = 18
        # Prefer static export for CoreML stability in the planned app scenario.
        # Keep legacy dynamic_axes behavior untouched.
        if args.dynamic_points:
            export_kwargs["dynamic_axes"] = dynamic_axes
        export_kwargs.update(
            {
                "dynamo": True,
                "report": args.dynamo_report,
                "verify": args.dynamo_verify,
                "profile": args.dynamo_profile,
                "artifacts_dir": args.dynamo_artifacts_dir,
            }
        )
    else:
        export_kwargs["dynamic_axes"] = dynamic_axes
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        export_model,
        (dummy, dummy_coords, dummy_labels),
        str(output_path),
        **export_kwargs,
    )

    print(f"Exported ONNX to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
