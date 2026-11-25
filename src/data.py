"""
Dataset loading and preprocessing module for SQuAD Question Answering.
"""

from functools import partial
from typing import Dict, Any, Optional, Tuple

try:
    from datasets import load_dataset, Dataset, DatasetDict
except ImportError:
    load_dataset = None
    Dataset = None
    DatasetDict = None


def prepare_train_features(
    examples: Dict[str, Any],
    tokenizer: Any,
    max_length: int = 384,
    doc_stride: int = 128,
) -> Dict[str, Any]:
    """
    Tokenizes question-context pairs with sliding windows and aligns character-level
    answer offsets to token-level start and end positions.
    """
    # Clean leading whitespace from questions
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize with sliding windows
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        # If no answers are given, label with [CLS] index
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Find context token span boundaries (sequence_id == 1)
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            # Check if answer is outside the current window
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Move token_start_index and token_end_index to two ends of answer
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def prepare_validation_features(
    examples: Dict[str, Any],
    tokenizer: Any,
    max_length: int = 384,
    doc_stride: int = 128,
) -> Dict[str, Any]:
    """
    Tokenizes validation question-context pairs with sliding windows, preserves example IDs,
    and masks non-context offset mappings with None for span post-processing.
    """
    examples["question"] = [q.lstrip() for q in examples["question"]]

    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1

        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set offsets outside context to None
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def load_and_preprocess_squad(
    tokenizer: Any,
    dataset_name: str = "rajpurkar/squad",
    train_samples: Optional[int] = 8000,
    val_samples: Optional[int] = 1000,
    max_length: int = 384,
    doc_stride: int = 128,
) -> Tuple[Any, Any, Any, Any]:
    """
    Loads SQuAD dataset, extracts subsets (if requested), and applies tokenization.

    Returns:
        raw_train: Raw train dataset / subset
        raw_val: Raw validation dataset / subset
        tokenized_train: Tokenized train features with start/end positions
        tokenized_val: Tokenized validation features with offset mappings and example IDs
    """
    if load_dataset is None:
        raise ImportError(
            "The 'datasets' package is required to load SQuAD. "
            "Please install dependencies with: pip install -r requirements.txt"
        )

    raw_datasets: DatasetDict = load_dataset(dataset_name)

    train_data = raw_datasets["train"]
    val_data = raw_datasets["validation"]

    if train_samples is not None and train_samples > 0 and train_samples < len(train_data):
        train_data = train_data.select(range(train_samples))

    if val_samples is not None and val_samples > 0 and val_samples < len(val_data):
        val_data = val_data.select(range(val_samples))

    train_fn = partial(
        prepare_train_features,
        tokenizer=tokenizer,
        max_length=max_length,
        doc_stride=doc_stride,
    )
    val_fn = partial(
        prepare_validation_features,
        tokenizer=tokenizer,
        max_length=max_length,
        doc_stride=doc_stride,
    )

    tokenized_train = train_data.map(
        train_fn,
        batched=True,
        remove_columns=train_data.column_names,
        desc="Tokenizing train dataset",
    )

    tokenized_val = val_data.map(
        val_fn,
        batched=True,
        remove_columns=val_data.column_names,
        desc="Tokenizing validation dataset",
    )

    return train_data, val_data, tokenized_train, tokenized_val
