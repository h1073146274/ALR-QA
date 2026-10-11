# ALR-QA

## 1. Environment and Models

```bash
cd ALR-QA-anonymous
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in the model names, API settings, input paths, and output paths in `.env`. The pipeline requires a Teacher API model, an Anchor Encoder, and a Base Generator. Set `ARCH=qwen` or `ARCH=phi` to select the generator entry point.

## 2. Split the Data

Split the chunks by document so that chunks from the same document do not appear in different data splits:

```bash
bash scripts/run_pipeline.sh split
```

The resulting train, validation, and test files are written to `SPLIT_OUT_DIR`.

## 3. Build the Training Data and Anchors

Use the Teacher API to create view-conditioned QA training data:

```bash
bash scripts/run_pipeline.sh dataset
```

Then encode the document, chunk, and view anchors:

```bash
bash scripts/run_pipeline.sh anchors
```

Main outputs:

```text
teacher_raw.jsonl
view_conditioned_qa.jsonl
view_conditioned_sft.jsonl
doc_anchor_emb.pt
chunk_anchor_emb.pt
view_anchor_emb.pt
view_conditioned_sft.with_anchor_index.jsonl
```

## 4. Train

```bash
bash scripts/run_pipeline.sh train
```

The architecture-specific entry points can also be called directly:

```bash
python train/qwen/train.py \
  --model_name_or_path "" \
  --train_jsonl "" \
  --doc_anchor_pt "" \
  --chunk_anchor_pt "" \
  --view_anchor_pt "" \
  --output_dir ""

python train/phi/train.py \
  --model_name_or_path "" \
  --train_jsonl "" \
  --doc_anchor_pt "" \
  --chunk_anchor_pt "" \
  --view_anchor_pt "" \
  --output_dir ""
```

Training saves the LoRA adapter, anchor projection heads, routing head, and view prototypes.

## 5. Generate QA

First, generate anchors for the input chunks:

```bash
bash scripts/run_pipeline.sh infer-anchors
```

Then generate the QA pairs:

```bash
bash scripts/run_pipeline.sh infer
```

The inference entry points can also be called directly:

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

Each output line contains one QA record:

```json
{"doc_id": "...", "chunk_id": "...", "selected_views": ["..."], "view": "...", "question": "...", "answer": "..."}
```

## 6. Evaluation

The evaluation code is located in [`evaluation/`](evaluation) and provides the following metrics:

- QRelScore (QRel)
- Source Hit@5
- Diverse@10 (D@10)
- Self-CosSim (SelfSim)
- RAGAS Faithfulness (FAI)
- RAGAS Answer Relevancy (REL)

Install the evaluation dependencies:

```bash
pip install -r evaluation/requirements.txt
```

QRel, Hit@5, D@10, and SelfSim require a sentence embedding model, a word-alignment model, and a causal language model:

```bash
python evaluation/eval_qa_metrics.py \
  --qa_file "" \
  --chunks_file "" \
  --out_dir "" \
  --sim_model "" \
  --word_model "" \
  --clm_model ""
```

FAI and REL require an OpenAI-compatible evaluation model and an embedding model:

```bash
OPENAI_API_KEY="" python evaluation/eval_ragas.py \
  --qa_file "" \
  --chunks_file "" \
  --out_dir "" \
  --llm_model "" \
  --api_base_url "" \
  --embedding_model ""
```

`qa_file` must be a JSON or JSONL file containing `chunk_id`, `question`, and `answer`. `chunks_file` must contain the chunk ID and source text. Results are written to the specified `out_dir`.
