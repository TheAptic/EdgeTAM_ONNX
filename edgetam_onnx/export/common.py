from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any


def resolve_sam2_root(explicit_root: str | None) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    default_root = project_root / "EdgeTAM" if (project_root / "EdgeTAM").exists() else project_root / "sam2"
    sam2_root = Path(explicit_root) if explicit_root else default_root
    if not sam2_root.exists():
        raise SystemExit(f"--sam2-root does not exist: {sam2_root}")
    return sam2_root


def ensure_export_deps() -> None:
    missing = []
    if importlib.util.find_spec("torch") is None:
        missing.append("torch")
    if importlib.util.find_spec("sam2") is None:
        missing.append("sam2")
    if missing:
        raise SystemExit(f"Missing export dependencies: {', '.join(missing)}")


def setup_sam2_import_path(sam2_root: Path) -> None:
    sys.path.insert(0, str(sam2_root))


def load_model(config: str, checkpoint: str, allow_backbone_download: bool = False):
    import torch
    from hydra import compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    _patch_prompt_encoder_for_onnx(torch)

    if not allow_backbone_download:
        try:
            timm_backbone_mod = importlib.import_module("sam2.modeling.backbones.timm")
            original_create_model = timm_backbone_mod.create_model

            def _offline_create_model(*model_args, **model_kwargs):
                model_kwargs["pretrained"] = False
                return original_create_model(*model_args, **model_kwargs)

            timm_backbone_mod.create_model = _offline_create_model
        except Exception:
            pass

    cfg = OmegaConf.load(config) if Path(config).exists() else compose(config_name=config)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
    if missing_keys:
        raise RuntimeError(f"Missing checkpoint keys: {missing_keys[:10]}")
    if unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected_keys[:10]}")
    model.eval()
    return model


def _patch_prompt_encoder_for_onnx(torch_module) -> None:
    try:
        prompt_mod = importlib.import_module("sam2.modeling.sam.prompt_encoder")
        PromptEncoder = prompt_mod.PromptEncoder
        if getattr(PromptEncoder, "_edgetam_onnx_patch_applied", False):
            return

        def _embed_points_onnx_safe(self, points, labels, pad):
            points = points + 0.5
            if pad:
                padding_point = torch_module.zeros((points.shape[0], 1, 2), device=points.device)
                padding_label = -torch_module.ones((labels.shape[0], 1), device=labels.device)
                points = torch_module.cat([points, padding_point], dim=1)
                labels = torch_module.cat([labels, padding_label], dim=1)

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
                sparse_embeddings = torch_module.cat(pieces, dim=1) if len(pieces) > 1 else pieces[0]
            else:
                sparse_embeddings = torch_module.zeros(
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


def sanitize_point_labels_for_onnx(point_labels, torch_module):
    ones = torch_module.ones_like(point_labels)
    zeros = torch_module.zeros_like(point_labels)
    neg_one = torch_module.full_like(point_labels, -1)
    return torch_module.where(
        point_labels == -1,
        neg_one,
        torch_module.where(point_labels > 0, ones, zeros),
    )


class EdgeTAMImageEncoderExport:
    def __init__(self, sam_model):
        import torch

        class _Impl(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, image):
                backbone_out = self.model.forward_image(image)
                backbone_fpn = backbone_out.get("backbone_fpn", [])
                vision_features = backbone_fpn[2]
                high_res_feat_0 = backbone_fpn[0]
                high_res_feat_1 = backbone_fpn[1]
                if self.model.directly_add_no_mem_embed:
                    b, c, h, w = vision_features.shape
                    vf = vision_features.flatten(2).permute(2, 0, 1)
                    vf = vf + self.model.no_mem_embed.squeeze(0)
                    vision_features = vf.permute(1, 2, 0).view(b, c, h, w)
                return vision_features, high_res_feat_0, high_res_feat_1

        self.module = _Impl(sam_model)


class EdgeTAMPromptEncoderExport:
    def __init__(self, sam_model):
        import torch

        class _Impl(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, point_coords, point_labels, box_coords, mask_input):
                point_labels = sanitize_point_labels_for_onnx(point_labels, torch)
                # Keep export/runtime stable first: ignore boxes in phase-1 static path.
                # Inputs are still present in the ONNX contract to keep shape/API compatibility.
                sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                    points=(point_coords, point_labels),
                    boxes=None,
                    masks=mask_input,
                )
                return sparse_embeddings, dense_embeddings

        self.module = _Impl(sam_model)


class EdgeTAMMaskDecoderExport:
    def __init__(self, sam_model):
        import torch

        class _Impl(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(
                self,
                image_embeddings,
                sparse_prompt_embeddings,
                dense_prompt_embeddings,
                high_res_feat_0,
                high_res_feat_1,
            ):
                sam_outputs = self.model.sam_mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_prompt_embeddings,
                    dense_prompt_embeddings=dense_prompt_embeddings,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=[high_res_feat_0, high_res_feat_1],
                )
                low_res_masks = sam_outputs[0][:, 0:1, :, :]
                iou_scores = sam_outputs[1][:, 0:1]
                final_masks = torch.nn.functional.interpolate(
                    low_res_masks,
                    size=(1024, 1024),
                    mode="bilinear",
                    align_corners=False,
                )
                return low_res_masks, final_masks, iou_scores

        self.module = _Impl(sam_model)
