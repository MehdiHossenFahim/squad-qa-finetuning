# squad-qa-finetuning

Production-ready extractive Question Answering (QA) pipeline fine-tuning `bert-base-uncased` on the Stanford Question Answering Dataset (SQuAD v1.1) using Hugging Face `transformers`, `datasets`, `evaluate`, and PyTorch.

---

## Architecture Overview

Extractive question answering models formulate the task as **span prediction**: given a concatenated sequence `[CLS] Question [SEP] Context [SEP]`, the model predicts the start and end token indices that demarcate the answer span inside the context passage.

```
Input Tokens:  [CLS]  Question  [SEP]  Context Token 1  ...  Context Token N  [SEP]
                  │       │       │            │                    │            │
             ┌────┴───────┴───────┴────────────┴────────────────────┴────────────┴────┐
             │                   BERT Base Encoder (12 layers, 768-d)                 │
             └────┬───────┬───────┬────────────┬────────────────────┬────────────┬────┘
                  │       │       │            │                    │            │
Hidden States:    h_0     h_1    ...          h_i                  h_j          h_L
                  │                            │                    │
             ┌────┴────────────────────────────┴────────────────────┴─────────────────┐
             │       Linear QA Head (in_features=768, out_features=2: Start/End)      │
             └─────────────────────────────────┬────────────────────┬─────────────────┘
                                               ▼                    ▼
                                         Start Logit           End Logit
```

- **Base Encoder:** `bert-base-uncased` (12 transformer encoder blocks, 768 hidden dimension, 12 attention heads, 110M parameters).
- **QA Prediction Head:** A 2-output linear classification layer on top of the contextual token representations producing unnormalized logits $s_i$ (start probability) and $e_j$ (end probability) for each sequence index.
- **Training Loss:** Cross-Entropy loss computed independently for the start position and end position, summed into the joint objective:
  $$\mathcal{L} = \mathcal{L}_{\text{start}} + \mathcal{L}_{\text{end}}$$

---

## SQuAD Preprocessing & Offset Mapping Strategy

SQuAD provides answer annotations as character-level start positions within the raw context string. Adapting this for subword transformer architectures requires:

1. **Sliding Window Chunking (`doc_stride=128`):**
   - Context passages often exceed BERT's maximum sequence length (`max_length=384`).
   - Inputs are tokenized with `truncation="only_second"` and `return_overflowing_tokens=True`.
   - Long documents are split into overlapping feature windows with a 128-token overlap (`doc_stride`), ensuring answer spans crossing boundaries remain captured.
2. **Character-to-Token Offset Alignment (`offset_mapping`):**
   - Fast tokenizers generate `offset_mapping` tuples `(start_char, end_char)` for each token relative to the original document string.
   - Non-context tokens (`[CLS]`, `[SEP]`, question tokens) are masked.
   - If the gold answer span is outside the current window chunk, labels default to the `[CLS]` token position (index 0).
   - If inside the window, token indices are adjusted to tightly enclose `[start_char, end_char]`.
3. **Logit Aggregation & Span Post-Processing:**
   - Raw model predictions output start and end logits for every token across all window chunks of an example.
   - For each validation example, candidate spans are generated from the top $N$ start and end indices (`n_best_size=20`).
   - Inadmissible spans are pruned (e.g., end < start, length > `max_answer_length=30`, or indices mapping outside the context).
   - Candidate spans are ranked by composite score $S = \text{logit}_{\text{start}} + \text{logit}_{\text{end}}$, and the highest-scoring character span is extracted.

---

## Training Setup & Hyperparameters

| Hyperparameter | Value | Description |
|---|---|---|
| **Base Model** | `bert-base-uncased` | Pretrained BERT checkpoint |
| **Dataset** | SQuAD v1.1 (`rajpurkar/squad`) | Extractive reading comprehension |
| **Train Samples** | 8,000 | Selected training subset for fast convergence |
| **Validation Samples** | 1,000 | Validation subset for evaluation |
| **Max Sequence Length** | 384 | Sequence token budget |
| **Document Stride** | 128 | Overlapping window stride |
| **Optimizer** | AdamW | Adaptive weight decay optimizer |
| **Learning Rate** | `3e-5` | Constant initial learning rate with linear warmup |
| **Weight Decay** | 0.01 | $L_2$ regularization penalty |
| **Batch Size** | 16 | Per-device train and evaluation batch size |
| **Epochs** | 3 | Total training epochs (1,521 global steps) |
| **Mixed Precision** | FP16 | Mixed precision GPU acceleration |
| **Final Training Loss** | `0.5918` | Training loss at step 1,521 |

---

## Validation Results

Evaluated on the SQuAD validation subset following span post-processing using official SQuAD v1.1 evaluation criteria:

| Metric | Score | Definition |
|---|---|---|
| **Exact Match (EM)** | **73.30%** | Percentage of predictions matching reference answers exactly (normalized) |
| **F1 Score** | **79.88%** | Macro-averaged token-level overlap between predicted and reference spans |

### Zero-Shot Inference Validation

| Question | Context Snippet | Predicted Answer |
|---|---|---|
| *Who developed the theory of relativity?* | "Albert Einstein developed the theory of relativity in the early 20th century." | `albert einstein` |
| *What is the capital of France?* | "France is a country in Western Europe. Its capital, Paris, is known for..." | `paris` |
| *When was the Hugging Face Transformers library released?* | "...Its Transformers library, first released in 2018, provides thousands of pretrained models..." | `2018` |

---

## Repository Structure

```
squad-qa-finetuning/
├── notebooks/
│   └── squad_qa_finetuning.ipynb   # Interactive walkthrough with execution metrics
├── src/
│   ├── __init__.py                 # Package initializer
│   ├── data.py                     # Sliding window tokenization & offset mapping
│   ├── evaluate.py                 # Span post-processing, EM & F1 evaluation
│   ├── inference.py                # Single-shot and interactive QA inference
│   └── train.py                    # Modular training pipeline with CLI arguments
├── .gitignore                      # Checkpoint and artifact exclusion
├── requirements.txt                # Pinned production dependencies
└── README.md                       # Architecture, reproduction, and usage guide
```

---

## Quickstart & Installation

### 1. Environment Setup

```bash
git clone https://github.com/<your-username>/squad-qa-finetuning.git
cd squad-qa-finetuning
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## How to Run

### 1. Model Training (`src/train.py`)

Fine-tune BERT on SQuAD with configurable arguments:

```bash
# Standard run (matching benchmark configuration: 8,000 train / 1,000 val)
python src/train.py \
    --model_name_or_path bert-base-uncased \
    --train_samples 8000 \
    --val_samples 1000 \
    --num_train_epochs 3 \
    --learning_rate 3e-5 \
    --batch_size 16 \
    --output_dir bert-squad-finetuned

# Full dataset training
python src/train.py \
    --model_name_or_path bert-base-uncased \
    --train_samples -1 \
    --val_samples -1 \
    --num_train_epochs 3 \
    --batch_size 16 \
    --output_dir results/bert-squad-full
```

### 2. Model Evaluation (`src/evaluate.py`)

Evaluate an existing fine-tuned checkpoint on the SQuAD validation split to compute Exact Match and F1:

```bash
python src/evaluate.py \
    --model_name_or_path bert-squad-finetuned \
    --val_samples 1000 \
    --batch_size 16 \
    --n_best_size 20 \
    --max_answer_length 30 \
    --output_file results/eval_metrics.json
```

### 3. Inference (`src/inference.py`)

#### Single-Shot Query:

```bash
python src/inference.py \
    --model_name_or_path bert-squad-finetuned \
    --question "When was the library released?" \
    --context "The Transformers library was first released in 2018 by Hugging Face."
```

#### Interactive CLI Session:

```bash
python src/inference.py \
    --model_name_or_path bert-squad-finetuned \
    --interactive
```

---

## License

This project is licensed under the MIT License.
