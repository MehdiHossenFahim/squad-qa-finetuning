"""
Evaluation and post-processing module for SQuAD Question Answering.
Computes Exact Match (EM) and F1 metrics on extractive span predictions.
"""

import argparse
import collections
import importlib
import json
import re
import string
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.data import prepare_validation_features
except ImportError:
    from data import prepare_validation_features


def _load_hf_evaluate():
    """
    Safely import Hugging Face evaluate package without shadowing from evaluate.py.
    """
    current_dir = str(Path(__file__).resolve().parent)
    original_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if Path(p).resolve() != Path(current_dir).resolve()]
        return importlib.import_module("evaluate")
    except ImportError:
        return None
    finally:
        sys.path = original_path


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact(a_gold: str, a_pred: str) -> int:
    """Computes exact match score between prediction and gold answer."""
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> float:
    """Computes token-level F1 score between prediction and gold answer."""
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return float(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def postprocess_qa_predictions(
    examples: Any,
    features: Any,
    raw_predictions: Tuple[Any, Any],
    n_best_size: int = 20,
    max_answer_length: int = 30,
) -> collections.OrderedDict:
    """
    Post-processes raw start/end logits into predicted text spans.
    Aggregates overlapping sliding windows per example and selects highest scoring valid span.
    """
    import numpy as np

    all_start_logits, all_end_logits = raw_predictions

    # Handle both HF Dataset and standard Python list/dict structures
    if isinstance(examples, (list, tuple)):
        example_ids = [ex["id"] for ex in examples]
    elif hasattr(examples, "column_names") or (isinstance(examples, dict) and "id" in examples):
        example_ids = examples["id"]
    else:
        example_ids = [ex["id"] for ex in examples]

    example_id_to_index = {k: i for i, k in enumerate(example_ids)}

    if isinstance(features, (list, tuple)):
        feature_example_ids = [f["example_id"] for f in features]
    elif hasattr(features, "column_names") or (isinstance(features, dict) and "example_id" in features):
        feature_example_ids = features["example_id"]
    else:
        feature_example_ids = [f["example_id"] for f in features]

    features_per_example = collections.defaultdict(list)
    for i, feature_example_id in enumerate(feature_example_ids):
        features_per_example[example_id_to_index[feature_example_id]].append(i)

    predictions = collections.OrderedDict()

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        context = example["context"]

        valid_answers = []
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            if (
                isinstance(features, dict)
                and "offset_mapping" in features
                and not hasattr(features, "column_names")
            ):
                offset_mapping = features["offset_mapping"][feature_index]
            else:
                offset_mapping = features[feature_index]["offset_mapping"]

            # Select top n_best_size start and end indices
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Ignore invalid offsets
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Ignore answers where end is before start or length exceeds threshold
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": float(start_logits[start_index] + end_logits[end_index]),
                            "text": context[start_char:end_char],
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = max(valid_answers, key=lambda x: x["score"])
        else:
            best_answer = {"text": "", "score": 0.0}

        predictions[example["id"]] = best_answer["text"]

    return predictions


def compute_squad_metrics(
    predictions: Dict[str, str],
    examples: Any,
) -> Dict[str, float]:
    """
    Computes Exact Match (EM) and F1 for SQuAD predictions against reference examples.
    Attempts to use Hugging Face evaluate, falling back to pure Python implementation.
    """
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    if isinstance(examples, (list, tuple)):
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    elif hasattr(examples, "column_names"):
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    elif isinstance(examples, dict) and "id" in examples and "answers" in examples:
        references = [
            {"id": examples["id"][i], "answers": examples["answers"][i]}
            for i in range(len(examples["id"]))
        ]
    else:
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]

    hf_evaluate = _load_hf_evaluate()
    if hf_evaluate is not None:
        try:
            squad_metric = hf_evaluate.load("squad")
            results = squad_metric.compute(predictions=formatted_predictions, references=references)
            return {
                "exact_match": round(float(results["exact_match"]), 2),
                "f1": round(float(results["f1"]), 2),
            }
        except Exception as e:
            print(f"Hugging Face evaluate encountered error: {e}. Falling back to internal metric computation.")

    # Native SQuAD v1.1 evaluation
    total = len(references)
    exact_match = 0.0
    f1 = 0.0

    for ref in references:
        pred_text = predictions.get(ref["id"], "")
        gold_answers = ref["answers"]["text"]
        if not gold_answers:
            continue
        exact_match += max(compute_exact(a, pred_text) for a in gold_answers)
        f1 += max(compute_f1(a, pred_text) for a in gold_answers)

    em_score = 100.0 * exact_match / total if total > 0 else 0.0
    f1_score = 100.0 * f1 / total if total > 0 else 0.0

    return {
        "exact_match": round(em_score, 2),
        "f1": round(f1_score, 2),
    }


def evaluate_model(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    dataset_name: str = "rajpurkar/squad",
    val_samples: Optional[int] = 1000,
    batch_size: int = 16,
    max_length: int = 384,
    doc_stride: int = 128,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Loads model, tokenizes validation set, produces predictions, and calculates EM / F1.
    """
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForQuestionAnswering,
            Trainer,
            TrainingArguments,
            default_data_collator,
        )
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            f"Missing required dependencies for evaluation: {e}. "
            "Please install dependencies with: pip install -r requirements.txt"
        )

    if tokenizer_name_or_path is None:
        tokenizer_name_or_path = model_name_or_path

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading tokenizer from: {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    print(f"Loading model from: {model_name_or_path}")
    model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
    model.to(device)

    print(f"Loading dataset: {dataset_name} (validation split)")
    dataset = load_dataset(dataset_name)
    val_subset = dataset["validation"]
    if val_samples is not None and val_samples > 0 and val_samples < len(val_subset):
        val_subset = val_subset.select(range(val_samples))

    print(f"Preprocessing validation features ({len(val_subset)} examples)...")
    tokenized_val = val_subset.map(
        lambda ex: prepare_validation_features(ex, tokenizer, max_length, doc_stride),
        batched=True,
        remove_columns=val_subset.column_names,
        desc="Tokenizing validation set",
    )

    trainer_eval_columns = tokenized_val.remove_columns(["example_id", "offset_mapping"])

    training_args = TrainingArguments(
        output_dir="eval_temp",
        per_device_eval_batch_size=batch_size,
        report_to="none",
        fp16=torch.cuda.is_available() and device.startswith("cuda"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=trainer_eval_columns,
        data_collator=default_data_collator,
    )

    print("Running model inference on validation features...")
    raw_predictions = trainer.predict(trainer_eval_columns)

    print("Post-processing predictions...")
    final_predictions = postprocess_qa_predictions(
        val_subset,
        tokenized_val,
        raw_predictions.predictions,
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
    )

    print("Computing SQuAD Exact Match and F1 metrics...")
    metrics = compute_squad_metrics(final_predictions, val_subset)

    print("=" * 40)
    print("Validation Results:")
    print(f"  Exact Match (EM): {metrics['exact_match']:.2f}")
    print(f"  F1 Score:         {metrics['f1']:.2f}")
    print("=" * 40)

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned QA model on SQuAD")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-squad-finetuned",
        help="Path to pretrained model directory or Hugging Face hub id",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="Path to tokenizer (defaults to model_name_or_path if None)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="rajpurkar/squad",
        help="Hugging Face dataset identifier",
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=1000,
        help="Number of validation samples to evaluate on (set to -1 or 0 for full set)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=384,
        help="Maximum sequence length after tokenization",
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="Sliding window stride when splitting long documents",
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        default=20,
        help="Number of best start/end logits to consider during postprocessing",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help="Maximum allowable length of predicted answer string",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save evaluation metrics as JSON",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device ('cuda', 'cpu', or None for auto)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    val_samples = None if (args.val_samples is None or args.val_samples <= 0) else args.val_samples

    metrics = evaluate_model(
        model_name_or_path=args.model_name_or_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        dataset_name=args.dataset_name,
        val_samples=val_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        device=args.device,
    )

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved evaluation metrics to: {output_path}")


if __name__ == "__main__":
    main()
