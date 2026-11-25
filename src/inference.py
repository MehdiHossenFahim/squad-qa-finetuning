"""
Inference script for SQuAD Question Answering.
Extracts answer spans for given question and context pairs using a fine-tuned model.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Any

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def answer_question(
    question: str,
    context: str,
    model: Any,
    tokenizer: Any,
    device: str = "cpu",
    max_length: int = 384,
) -> str:
    """
    Extracts answer span for a single question and context pair.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for inference. Install via pip install torch")

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=max_length,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    start_index = torch.argmax(outputs.start_logits, dim=-1).item()
    end_index = torch.argmax(outputs.end_logits, dim=-1).item()

    input_ids = inputs["input_ids"][0]
    if end_index >= start_index:
        answer_tokens = input_ids[start_index : end_index + 1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    else:
        answer = ""

    return answer.strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Run extractive QA inference with a fine-tuned model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-squad-finetuned",
        help="Path to fine-tuned model checkpoint or Hugging Face ID",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question to ask",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Context paragraph containing the answer",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive CLI session",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda' or 'cpu')",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    except ImportError as e:
        raise ImportError(
            f"Missing required inference dependencies: {e}. "
            "Please install dependencies with: pip install -r requirements.txt"
        )

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model and tokenizer from: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.eval()

    if args.interactive:
        print("\n=== Interactive Extractive QA (Type 'quit' or 'exit' to stop) ===")
        context = ""
        while True:
            ctx_input = input("\nEnter Context (or press Enter to keep previous): ").strip()
            if ctx_input.lower() in ["quit", "exit"]:
                break
            if ctx_input:
                context = ctx_input

            if not context:
                print("Context cannot be empty. Please provide context text.")
                continue

            question = input("Enter Question: ").strip()
            if question.lower() in ["quit", "exit"]:
                break
            if not question:
                continue

            ans = answer_question(question, context, model, tokenizer, device=device)
            print(f"\n[Answer]: {ans}")

    elif args.question and args.context:
        ans = answer_question(args.question, args.context, model, tokenizer, device=device)
        print("\n" + "=" * 40)
        print(f"Question: {args.question}")
        print(f"Context:  {args.context}")
        print(f"Answer:   {ans}")
        print("=" * 40)
    else:
        # Default showcase examples from notebook
        showcase = [
            (
                "Who developed the theory of relativity?",
                "Albert Einstein developed the theory of relativity in the early 20th century.",
            ),
            (
                "What is the capital of France?",
                "France is a country in Western Europe. Its capital, Paris, is known for the Eiffel Tower and the Louvre Museum.",
            ),
            (
                "When was the Hugging Face Transformers library released?",
                (
                    "Hugging Face is a company that develops tools for building applications using machine learning. "
                    "Its Transformers library, first released in 2018, provides thousands of pretrained models "
                    "for tasks such as text classification, question answering, and translation."
                ),
            ),
        ]
        print("\nRunning default showcase examples:")
        for q, c in showcase:
            ans = answer_question(q, c, model, tokenizer, device=device)
            print("-" * 50)
            print(f"Q: {q}")
            print(f"A: {ans}")


if __name__ == "__main__":
    main()
