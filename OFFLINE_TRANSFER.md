# Offline Transfer Guide

This guide is for preparing assets on this computer and transferring them to the RTX 4090D computer.

## Network Strategy

Do not rely on `huggingface-cli` or `hf-mirror.com` for the 4090D computer.

Use this workflow instead:

1. Download models through ModelScope or another accessible source.
2. Register those local model directories into this project with `scripts/download_models.py`.
3. Transfer the whole project or `offline_bundle/model_cache` to the 4090D computer.
4. Run the 4090D computer with local model paths only.

For Python packages, mainland pip mirrors are still useful:

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

## Prepare a Small Transfer Bundle

This bundle includes code, configs, seed rules, and scripts. It excludes large model cache.

```bash
cd greenwashing_detector
python scripts/make_transfer_bundle.py
```

The output is:

```text
../greenwashing_detector_transfer.tar.gz
```

## Transfer the ClimateBERT Detector

For `climatebert/environmental-claims`, use the full offline package prepared in the parent folder:

```text
../climatebert_environmental_claims_offline_full.tar.gz
```

This full package includes both `model.safetensors` and `pytorch_model.bin`, plus tokenizer and config files. After copying it to the 4090D computer, extract it from inside the project directory:

```bash
cd greenwashing_detector
mkdir -p offline_bundle/model_cache
tar -xzf ../climatebert_environmental_claims_offline_full.tar.gz -C offline_bundle/model_cache
```

The detector should then exist at:

```text
offline_bundle/model_cache/snapshots/climatebert__environmental-claims/
```

## Prepare Wheelhouse

This downloads Python wheels for offline installation:

```bash
cd greenwashing_detector
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash scripts/download_wheelhouse.sh
```

Transfer `offline_bundle/wheelhouse/` together with the project.

On the 4090D computer:

```bash
pip install --no-index --find-links offline_bundle/wheelhouse -r requirements.txt
```

PyTorch CUDA wheels can be large and version-sensitive. If the 4090D machine already has a working CUDA/PyTorch environment, use that environment and only install the remaining requirements.

## Register ModelScope / Local Model Directories

Example local paths, adjust them to your actual ModelScope download locations:

```bash
cd greenwashing_detector
python scripts/download_models.py \
  --generator-local /path/to/Qwen3-8B \
  --embedding-local /path/to/Qwen3-Embedding-0.6B \
  --detector-local /path/to/climatebert-environmental-claims
```

Prompt Guard is optional. Since you already downloaded it on the 4090D computer, register it separately:

```bash
python scripts/download_models.py --prompt-guard-local /path/to/Prompt-Guard-86M
```

If you downloaded a different compatible variant, either pass the model path directly when running safety evaluation, or update `optional_models.prompt_guard` to the same id you use here:

```bash
python scripts/download_models.py \
  --prompt-guard-id LLM-Research/Llama-Prompt-Guard-2-86M \
  --prompt-guard-local /path/to/Llama-Prompt-Guard-2-86M

python -m src.eval.eval_safety \
  --config configs/default.yaml \
  --guard-mode model \
  --prompt-guard-model /path/to/Llama-Prompt-Guard-2-86M
```

By default this creates symlinks under:

```text
offline_bundle/model_cache/snapshots/
```

If you want a self-contained transfer folder, use `--copy` instead of symlinks:

```bash
python scripts/download_models.py \
  --copy \
  --generator-local /path/to/Qwen3-8B \
  --embedding-local /path/to/Qwen3-Embedding-0.6B \
  --detector-local /path/to/climatebert-environmental-claims
```

With `safety.prompt_guard_mode: auto`, the system uses model-backed Prompt Guard after it is registered. If it is not registered, `eval_safety` and `analyze_claim` fall back to the rule-based guard.

## Prepare Guidance Pages

The project includes seed rules, so it can run without downloading official pages. If network allows:

```bash
python scripts/prepare_offline_assets.py --download-guidelines
```

If this fails, keep using the built-in seed rules and cite official sources manually in the final report.

## Environmental Claims Dataset

The detector evaluation uses the `climatebert/environmental_claims` dataset. To avoid Hugging Face access on the 4090D computer, the project includes local JSONL files under:

```text
data/raw/environmental_claims/
```

The loader reads these local files before trying any online `load_dataset(...)` call. If you only need to transfer the dataset supplement, use:

```text
../environmental_claims_dataset_offline.tar.gz
```

Extract it from inside the project directory:

```bash
cd greenwashing_detector
tar -xzf ../environmental_claims_dataset_offline.tar.gz
```

## On the 4090D Computer

After extraction:

```bash
cd greenwashing_detector
conda create -n greenwashing python=3.10 -y
conda activate greenwashing
pip install --no-index --find-links offline_bundle/wheelhouse -r requirements.txt
```

Then build local corpus/index:

```bash
python -m src.data.load_environmental_claims --config configs/default.yaml
python -m src.data.build_rule_corpus --config configs/default.yaml
python -m src.data.build_eval_claims --config configs/default.yaml
python -m src.index.build_faiss --config configs/default.yaml
```

Smoke test without LLM:

```bash
python -m src.generation.analyze_claim \
  --config configs/default.yaml \
  --claim "Our bottle is 100% eco-friendly and carbon neutral." \
  --product-category "packaging" \
  --jurisdiction "general" \
  --no-llm
```

Full test with Qwen:

```bash
python -m src.generation.analyze_claim \
  --config configs/default.yaml \
  --claim "Our bottle is 100% eco-friendly and carbon neutral." \
  --product-category "packaging" \
  --jurisdiction "general"
```

Launch demo:

```bash
python -m src.app.gradio_app --config configs/default.yaml --host 0.0.0.0 --port 7860
```
