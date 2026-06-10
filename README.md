# Greenwashing Detector

Retrieval-augmented green marketing claim risk detection and responsible rewriting system.

This project is prepared for the coursework prototype. It is designed so the current computer can prepare code/resources, while the RTX 4090D computer only needs to run indexing, evaluation, and demo commands.

## What It Does

Input:

```text
Our bottle is 100% eco-friendly and carbon neutral.
```

Output:

- Environmental-claim detection
- Greenwashing risk level and categories
- Retrieved FTC/CMA/EU/ASA guidance snippets
- Citation-grounded reasoning
- Responsible rewrite
- Consumer-facing explanation
- Ethics and compliance notes
- Optional Prompt Guard input-safety check

## Quick Start on the 4090D Computer

If the 4090D computer cannot access Hugging Face, use local model directories downloaded from ModelScope or transferred from another computer. Register them before running:

```bash
python scripts/download_models.py \
  --generator-local /path/to/Qwen3-8B \
  --embedding-local /path/to/Qwen3-Embedding-0.6B \
  --detector-local /path/to/climatebert-environmental-claims
```

If you already downloaded Prompt Guard on the 4090D computer, register it as the optional safety model:

```bash
python scripts/download_models.py --prompt-guard-local /path/to/Prompt-Guard-86M
```

For pip, you can use a mainland mirror:

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

```bash
conda create -n greenwashing python=3.10 -y
conda activate greenwashing
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you transfer an offline wheelhouse:

```bash
pip install --no-index --find-links offline_bundle/wheelhouse -r requirements.txt
```

Build local guidance corpus and FAISS index:

```bash
python -m src.data.build_rule_corpus --config configs/default.yaml
python -m src.index.build_faiss --config configs/default.yaml
```

Run smoke analysis:

```bash
python -m src.generation.analyze_claim \
  --config configs/default.yaml \
  --claim "Our bottle is 100% eco-friendly and carbon neutral." \
  --product-category "packaging" \
  --jurisdiction "general"
```

Launch demo:

```bash
python -m src.app.gradio_app --config configs/default.yaml
```

Run evaluations after preparing data:

```bash
python -m src.data.load_environmental_claims --config configs/default.yaml
python -m src.data.build_rule_corpus --config configs/default.yaml
python -m src.index.build_faiss --config configs/default.yaml
python -m src.eval.eval_detector --config configs/default.yaml
python -m src.eval.eval_retrieval --config configs/default.yaml
python -m src.eval.eval_risk_classification --config configs/default.yaml
python -m src.eval.eval_rewrite_quality --config configs/default.yaml
python -m src.eval.eval_safety --config configs/default.yaml
python -m src.eval.make_figures --config configs/default.yaml
```

Prompt Guard is optional but useful for the ethics/safety part of the experiment. With `safety.prompt_guard_mode: auto`, `eval_safety` and `analyze_claim` use the local model when it has been registered; otherwise they fall back to the rule-based guard.

To force the model-backed safety check:

```bash
python -m src.eval.eval_safety --config configs/default.yaml --guard-mode model
```

Or pass a downloaded local directory directly:

```bash
python -m src.eval.eval_safety \
  --config configs/default.yaml \
  --guard-mode model \
  --prompt-guard-model /path/to/Prompt-Guard-86M
```

## Offline Preparation on This Computer

Prepare guidance text and optional wheelhouse:

```bash
python scripts/prepare_offline_assets.py --config configs/default.yaml --download-guidelines
python scripts/make_transfer_bundle.py
```

If this computer can access Hugging Face directly, optional direct download is still available:

```bash
python scripts/download_models.py --config configs/default.yaml --download-from-hf generator,embedding,detector
```

For ModelScope-downloaded models, use `--*-local` paths instead. This avoids Hugging Face access on the 4090D computer.

The `climatebert/environmental_claims` detector-evaluation dataset is also available locally under `data/raw/environmental_claims/`, so `eval_detector` does not need to contact Hugging Face.

## Project Layout

```text
configs/                  YAML configuration
data/rules/               Official green-claim guidance corpus
data/processed/           Processed claims and chunks
indexes/                  FAISS index and metadata
src/data/                 Dataset and corpus builders
src/index/                FAISS indexing
src/retrieval/            Search and reranking
src/models/               ClimateBERT and Qwen loading
src/generation/           Prompting, JSON schema, claim analysis
src/safety/               Citation audit and ethics checks
src/eval/                 Evaluation scripts
src/app/                  Gradio UI
scripts/                  Offline transfer helpers
```

## Notes

- The system outputs risk indications, not final legal conclusions.
- If evidence is insufficient, the output should say so explicitly.
- All risk categories should be grounded in citations from retrieved guidance or evidence.
