from __future__ import annotations

from dataclasses import dataclass

import onnxruntime as ort


@dataclass
class SplitOrtSessions:
    image_encoder: ort.InferenceSession
    prompt_encoder: ort.InferenceSession
    mask_decoder: ort.InferenceSession


def create_split_sessions(
    image_encoder_path: str,
    prompt_encoder_path: str,
    mask_decoder_path: str,
    providers: list[str] | None = None,
) -> SplitOrtSessions:
    p = providers or ["CPUExecutionProvider"]
    return SplitOrtSessions(
        image_encoder=ort.InferenceSession(image_encoder_path, providers=p),
        prompt_encoder=ort.InferenceSession(prompt_encoder_path, providers=p),
        mask_decoder=ort.InferenceSession(mask_decoder_path, providers=p),
    )
