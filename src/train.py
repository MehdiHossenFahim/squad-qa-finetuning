"""
Model fine-tuning script for Extractive Question Answering on SQuAD v1.1.
Supports configurable hyperparameters via CLI arguments.
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.data import load_and_preprocess_squad
    from src.evaluate import postprocess_qa_predictions, compute_squad_metrics
except ImportError:
    from data import load_and_preprocess_squad
    from evaluate import postprocess_qa_predictions, compute_squad_metrics


def set_reproducibility(seed: int = 42):
    """Sets random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        from transformers import set_seed
        set_seed(seed)
    except ImportError:
        pass


def train(
    model_name_or_path: str = "bert-base-uncased",
    dataset_name: str = "rajpurkar/squad",
    output_dir: str = "bert-squad-finetuned",
    train_samples: Optional[int] = 8000,
    val_samples: Optional[int] = 1000,
    max_length: int = 384,
    doc_stride: int = 128,
    learning_rate: float = 3e-5,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 16,
    num_train_epochs: int = 3,
    weight_decay: float = 0.01,
    seed: int = 42,
    fp16: Optional[bool] = None,
    save_strategy: str = "no",
    eval_strategy: str = "epoch",
    do_eval: bool = True,
    n_best_size: int = 20,
    max_answer_length: int = 30,
):
    """
    Fine-tunes a transformer model on SQuAD and optionally evaluates EM / F1.
    """
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForQuestionAnswering,
            TrainingArguments,
            Trainer,
            default_data_collator,
        )
    except ImportError as e:
        raise ImportError(
            f"Missing required training dependencies: {e}. "
            "Please install dependencies with: pip install -r requirements.txt"
        )

    set_reproducibility(seed)

    if fp16 is None:
        fp16 = torch.cuda.is_available()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | CUDA Available: {torch.cuda.is_available()}")

    # 1. Load Tokenizer
    print(f"Loading tokenizer: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # 2. Load and Preprocess Dataset
    print(f"Loading and preprocessing dataset: {dataset_name}")
    raw_train, raw_val, tokenized_train, tokenized_val = load_and_preprocess_squad(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        train_samples=train_samples,
        val_samples=val_samples,
        max_length=max_length,
        doc_stride=doc_stride,
    )

    print(f"Train samples (raw): {len(raw_train)} | Tokenized features: {len(tokenized_train)}")
    print(f"Validation samples (raw): {len(raw_val)} | Tokenized features: {len(tokenized_val)}")

    # 3. Load QA Model
    print(f"Loading model: {model_name_or_path}")
    model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
    model.to(device)

    # 4. Prepare Trainer arguments
    training_kwargs = {
        "output_dir": output_dir,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "num_train_epochs": num_train_epochs,
        "weight_decay": weight_decay,
        "fp16": fp16 and torch.cuda.is_available(),
        "save_strategy": save_strategy,
        "report_to": "none",
        "seed": seed,
    }

    try:
        training_args = TrainingArguments(eval_strategy=eval_strategy, **training_kwargs)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy=eval_strategy, **training_kwargs)

    # Remove non-model columns for loss evaluation
    trainer_eval_columns = tokenized_val.remove_columns(["example_id", "offset_mapping"])

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train,
        "eval_dataset": trainer_eval_columns,
        "data_collator": default_data_collator,
    }

    try:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)

    # 5. Train Model
    print("Starting model fine-tuning...")
    train_result = trainer.train()
    print("Training completed.")
    print(f"Global Steps: {train_result.global_step}")
    print(f"Training Loss: {train_result.training_loss:.4f}")

    # 6. Save Model & Tokenizer
    print(f"Saving fine-tuned model and tokenizer to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 7. Post-Training Validation Evaluation (Exact Match & F1)
    if do_eval:
        print("\nRunning post-training evaluation (EM / F1)...")
        raw_predictions = trainer.predict(trainer_eval_columns)
        final_predictions = postprocess_qa_predictions(
            raw_val,
            tokenized_val,
            raw_predictions.predictions,
            n_best_size=n_best_size,
            max_answer_length=max_answer_length,
        )
        metrics = compute_squad_metrics(final_predictions, raw_val)
        print("=" * 40)
        print("Validation Results:")
        print(f"  Exact Match (EM): {metrics['exact_match']:.2f}")
        print(f"  F1 Score:         {metrics['f1']:.2f}")
        print("=" * 40)
        return metrics

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT on SQuAD v1.1 for Extractive QA")

    # Model & Data paths
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased", help="Base pretrained model")
    parser.add_argument("--dataset_name", type=str, default="rajpurkar/squad", help="Dataset name on Hugging Face hub")
    parser.add_argument("--output_dir", type=str, default="bert-squad-finetuned", help="Directory to save model")

    # Data sizing
    parser.add_argument("--train_samples", type=int, default=8000, help="Number of training samples (use -1 for full)")
    parser.add_argument("--val_samples", type=int, default=1000, help="Number of validation samples (use -1 for full)")

    # Preprocessing
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--doc_stride", type=int, default=128, help="Sliding window stride length")

    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", "--batch_size", type=int, default=16, help="Train batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Eval batch size")
    parser.add_argument("--num_train_epochs", "--epochs", type=int, default=3, help="Total training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--fp16", action="store_true", default=None, help="Enable mixed precision fp16")
    parser.add_argument("--no_fp16", action="store_false", dest="fp16", help="Disable mixed precision fp16")
    parser.add_argument("--save_strategy", type=str, default="no", choices=["no", "epoch", "steps"])
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["no", "epoch", "steps"])

    # Evaluation
    parser.add_argument("--no_eval", action="store_false", dest="do_eval", help="Skip post-training EM/F1 evaluation")
    parser.add_argument("--n_best_size", type=int, default=20, help="N best logits for answer candidates")
    parser.add_argument("--max_answer_length", type=int, default=30, help="Max length for answer span")

    return parser.parse_args()


def main():
    args = parse_args()
    train_samples = None if (args.train_samples is None or args.train_samples <= 0) else args.train_samples
    val_samples = None if (args.val_samples is None or args.val_samples <= 0) else args.val_samples

    train(
        model_name_or_path=args.model_name_or_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        train_samples=train_samples,
        val_samples=val_samples,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        seed=args.seed,
        fp16=args.fp16,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        do_eval=args.do_eval,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
    )


if __name__ == "__main__":
    main()
