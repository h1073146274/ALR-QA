# Anchoring Latent Representations for Controlled Long-Document Question-Answer Generation

## 1. 🧰 Set Up the Environment

```bash
cd ""
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 🧠 Prepare the Models

Reproducing this pipeline requires three types of models.

Teacher API model: used to build the training data.

Anchor encoder: used to precompute document/chunk/view anchors offline, such as BGE-M3.

Base generator model: used for LoRA training and final inference. This repository provides two entry points:

```text
ARCH=qwen  -> train/qwen/train.py, inference/qwen/infer.py
ARCH=phi   -> train/phi/train.py, inference/phi/infer.py
```

Training will produce a LoRA adapter, which is loaded during inference through `LORA_DIR`.

## 3. 🔐 Fill in `.env`

The `.env` file in this repository is empty by default. It only keeps credentials, model names/paths, and the main output directory:

```text
API_KEY=
API_BASE_URL=
TEACHER_MODEL=
ENCODER_MODEL=
BASE_MODEL=
OUTPUT_DIR=
```

For inference, also fill in:

```text
LORA_DIR=
```

## 4. ✂️ Split the Chunk Data

Split chunks by document with an 8:1:1 train/validation/test ratio, so chunks from the same document do not appear in different splits.

```bash
bash scripts/run_pipeline.sh split
```

Outputs are written to `SPLIT_OUT_DIR`:

```text
{name}_train.jsonl
{name}_val.jsonl
{name}_test.jsonl
{name}_split_manifest.json
```

## 5. 📦 Build the View-Conditioned Training Dataset

```bash
bash scripts/run_pipeline.sh dataset
```

This step reads the training split and calls the teacher API to generate view-conditioned QA data for SFT.

Outputs are written to `DATASET_OUT_DIR`:

```text
teacher_raw.jsonl
view_conditioned_qa.jsonl
view_conditioned_sft.jsonl
```

## 6. 🧭 Build Offline Anchors

```bash
bash scripts/run_pipeline.sh anchors
```

Outputs are written to `ANCHOR_OUT_DIR`:

```text
doc_anchor_emb.pt
chunk_anchor_emb.pt
view_anchor_emb.pt
view_conditioned_sft.with_anchor_index.jsonl
```

## 7. 🚀 Train

After setting the required training variables:

```bash
bash scripts/run_pipeline.sh train
```

You can also run each architecture directly:

```bash
python train/qwen/train.py \
  --model_name_or_path "" \
  --train_jsonl "" \
  --doc_anchor_pt "" \
  --chunk_anchor_pt "" \
  --output_dir ""

python train/phi/train.py \
  --model_name_or_path "" \
  --train_jsonl "" \
  --doc_anchor_pt "" \
  --chunk_anchor_pt "" \
  --output_dir ""
```

## 8. 🔎 Generate QA with Inference

The test split also needs encoded anchors first:

```bash
bash scripts/run_pipeline.sh infer-anchors
```

Then run inference:

```bash
bash scripts/run_pipeline.sh infer
```

Direct entry points:

```bash
python inference/qwen/infer.py \
  --base_model "" \
  --lora_dir "" \
  --doc_anchor_pt "" \
  --chunk_anchor_pt "" \
  --input_jsonl "" \
  --output_jsonl ""

python inference/phi/infer.py \
  --base_model "" \
  --lora_dir "" \
  --doc_anchor_pt "" \
  --chunk_anchor_pt "" \
  --input_jsonl "" \
  --output_jsonl ""
```

Each inference output line is one QA record:

```json
{"doc_id": "...", "chunk_id": "...", "selected_views": ["..."], "view": "...", "question": "...", "answer": "..."}
```
