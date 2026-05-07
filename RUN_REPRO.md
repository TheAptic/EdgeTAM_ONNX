# Reproduce Split ONNX Results

## Prerequisites (required files not included in this repo)

Provide these locally before running:

1. Upstream source checkout:
   - `./EdgeTAM/`
2. Model checkpoint:
   - `./model/edgetam.pt`
3. Model config:
   - `./model/edgetam.yaml`

Preflight check:

```bash
test -d EdgeTAM && test -f model/edgetam.pt && test -f model/edgetam.yaml && echo "prereqs ok"
```

## 0) Environment setup (fresh clone / after cleanup)

```bash
cd /path/to/edgetam

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements-onnx.txt
pip install numpy opencv-python pillow torch onnx onnxruntime hydra-core omegaconf
```

Quick sanity checks:

```bash
python -c "import numpy, cv2, onnxruntime, torch; print('deps ok')"
PYTHONPATH=. python -m unittest tests.test_split_onnx_scaffold tests.test_onnx_parity_harness
```

## 1) Export split ONNX models

```bash
PYTHONPATH=. uv run python edgetam_onnx/export/export_image_encoder.py
PYTHONPATH=. uv run python edgetam_onnx/export/export_prompt_encoder.py --max-points 4
PYTHONPATH=. uv run python edgetam_onnx/export/export_mask_decoder.py --max-points 4
```

## 2) Generate masks from point examples (no overwrite)

```bash
PYTHONPATH=. uv run python scripts/generate_split_masks.py \
  --image point_mask_examples/reference_image.JPG \
  --points-dir point_mask_examples \
  --size 1024 \
  --max-points 4
```

This creates new files like:
- `point_mask_examples/bad1_split_new.png`
- `point_mask_examples/bad4-points_split_new.png`
- `point_mask_examples/bad5-points-*_split_new.png`
- `point_mask_examples/bad6-points_split_new.png`

## 3) Stage parity check vs PyTorch

```bash
PYTHONPATH=. uv run python edgetam_onnx/validate/compare_pytorch_onnx.py \
  --image point_mask_examples/reference_image.JPG \
  --points-file point_mask_examples/bad6-points.txt \
  --max-points 4 \
  --precision fp32 \
  --out artifacts/onnx_split/compare_bad6_fp32.json
```

## 4) Benchmark split ONNX vs PyTorch

```bash
PYTHONPATH=. uv run python scripts/benchmark_split_vs_pytorch.py \
  --image point_mask_examples/reference_image.JPG \
  --points-file point_mask_examples/bad6-points.txt \
  --size 1024 \
  --max-points 4 \
  --warmup 10 \
  --runs 50 \
  --out artifacts/benchmarks/split_vs_pytorch_bad6.json
```
