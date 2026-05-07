# EdgeTAM ONNX Split Export

This repository contains scripts and documentation for exporting EdgeTAM to a split ONNX pipeline (`image_encoder`, `prompt_encoder`, `mask_decoder`) and reproducing parity/benchmark results.

## Prerequisites (Required)

This repository does **not** include the upstream `EdgeTAM` source checkout or model checkpoints.

To reproduce results, you must provide:

1. Local `EdgeTAM` checkout at repo root:
   - `./EdgeTAM/`
2. EdgeTAM checkpoint file (for example):
   - `./model/edgetam.pt` (or a path you pass to export/validation scripts)
3. EdgeTAM config file (for example):
   - `./model/edgetam.yaml` (or a path you pass to export/validation scripts)

All split ONNX export and PyTorch parity scripts depend on these assets.

## EdgeTAM Example

![EdgeTAM example](docs/assets/edgetam_example.gif)

Source video: `EdgeTAM_example.mov`

## Reproduction

See `RUN_REPRO.md` for full setup and reproducible commands.
