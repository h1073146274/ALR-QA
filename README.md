# Anchoring Latent Representations for Controlled Long-Document Question–Answer Generation


## 1. 创建环境

```bash
cd ""
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 准备模型

复现需要三类模型。

Teacher API model：用于构建训练数据。它需要支持 OpenAI-compatible chat completions。你可以用 DeepSeek、GPT 类接口，或任何兼容的本地服务。

Anchor encoder：用于离线生成 document/chunk/view anchors，比如 BGE-M3。

Base generator model：用于 LoRA 训练和最终推理。仓库提供两种入口：

```text
ARCH=qwen  -> train/qwen/train.py, inference/qwen/infer.py
ARCH=phi   -> train/phi/train.py, inference/phi/infer.py
```

训练后还会得到一个 LoRA adapter，推理时通过 `LORA_DIR` 加载。

## 3. 填 `.env`

仓库里的 `.env` 默认全是空值，只放凭据、模型和主要输出目录：

```text
API_KEY=
API_BASE_URL=
TEACHER_MODEL=
ENCODER_MODEL=
BASE_MODEL=
OUTPUT_DIR=
```

推理时补：

```text
LORA_DIR=
```

数据路径、split 输出目录、anchor 输出目录、训练超参和推理输出路径建议在运行命令时传入环境变量或直接用脚本参数给出。

## 4. 切分 chunk 数据

按文档级别做 8:1:1，避免同一篇文档的 chunk 跑到不同 split 里。

```bash
bash scripts/run_pipeline.sh split
```

输出在 `SPLIT_OUT_DIR`：

```text
{name}_train.jsonl
{name}_val.jsonl
{name}_test.jsonl
{name}_split_manifest.json
```

## 5. 构建 View-conditioned Training Dataset

```bash
bash scripts/run_pipeline.sh dataset
```

这一步会调用 teacher API：

1. 根据 document anchor、chunk anchor、chunk body 选择最多 K 个 view。
2. 对每个 view 生成一个问题。
3. 用同一个 view、问题和 chunk body 生成答案。
4. 做 acceptance filtering。

输出在 `DATASET_OUT_DIR`：

```text
teacher_raw.jsonl
view_conditioned_qa.jsonl
view_conditioned_sft.jsonl
```

训练使用 `view_conditioned_sft.jsonl`。

## 6. 构建离线锚

```bash
bash scripts/run_pipeline.sh anchors
```

输出在 `ANCHOR_OUT_DIR`：

```text
doc_anchor_emb.pt
chunk_anchor_emb.pt
view_anchor_emb.pt
view_conditioned_sft.with_anchor_index.jsonl
```

## 7. 训练

填好训练相关变量后：

```bash
bash scripts/run_pipeline.sh train
```

也可以单独跑：

```bash
python train/qwen/train.py --model_name_or_path "" --train_jsonl "" --doc_anchor_pt "" --chunk_anchor_pt "" --output_dir ""
python train/phi/train.py --model_name_or_path "" --train_jsonl "" --doc_anchor_pt "" --chunk_anchor_pt "" --output_dir ""
```

## 8. 推理生成 QA

测试 split 也要先编码 anchors：

```bash
bash scripts/run_pipeline.sh infer-anchors
```

然后推理：

```bash
bash scripts/run_pipeline.sh infer
```

单独入口：

```bash
python inference/qwen/infer.py --base_model "" --lora_dir "" --doc_anchor_pt "" --chunk_anchor_pt "" --input_jsonl "" --output_jsonl ""
python inference/phi/infer.py --base_model "" --lora_dir "" --doc_anchor_pt "" --chunk_anchor_pt "" --input_jsonl "" --output_jsonl ""
```

推理输出每行是一条 QA：

```json
{"doc_id": "...", "chunk_id": "...", "selected_views": ["..."], "view": "...", "question": "...", "answer": "..."}
```
