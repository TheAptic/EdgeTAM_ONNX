# EdgeTAM ONNX Split Pipeline

## Model Files
Expected split model paths:
- `models/image_encoder.onnx`
- `models/prompt_encoder.onnx`
- `models/mask_decoder.onnx`

## Export Scripts
- `export/export_image_encoder.py`
- `export/export_prompt_encoder.py`
- `export/export_mask_decoder.py`

## Validation
- Stage parity: `validate/compare_pytorch_onnx.py`
- Prompt/mask parity harness: `scripts/onnx_parity_harness.py`

## Runtime Scaffolding
- Session creation: `runtime/ort_session.py`
- Tracker state placeholder: `runtime/tracker_state.py`

## Current Constraints
- Fixed prompt slots (`max_points=4` in current exported artifacts)
- `box_coords` exists in interface but current stabilized prompt export ignores boxes
