#!/usr/bin/env python3

from collections import Counter

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi


SOURCE_REPO = "sinhala-nlp/SOLD"
TARGET_REPO = "Sinhala-NLP/SiMTEB-SOLD"

LABEL_MAP = {
    "NOT": 0,
    "OFF": 1,
}


def convert_example(example):
    text = example["text"].strip()
    label = example["label"]

    if label not in LABEL_MAP:
        raise ValueError(f"Unexpected SOLD label: {label!r}")

    return {
        "text": text,
        "label": LABEL_MAP[label],
    }


def check_duplicates(dataset):
    """
    Report exact text duplicates both within and across splits.
    Do not automatically delete anything.
    """

    train_texts = dataset["train"]["text"]
    test_texts = dataset["test"]["text"]

    train_set = set(train_texts)
    test_set = set(test_texts)

    train_duplicates = len(train_texts) - len(train_set)
    test_duplicates = len(test_texts) - len(test_set)
    cross_split = train_set.intersection(test_set)

    print("\nDuplicate audit")
    print("----------------")
    print(f"Train exact duplicates: {train_duplicates}")
    print(f"Test exact duplicates:  {test_duplicates}")
    print(f"Train/test overlap:     {len(cross_split)}")

    if cross_split:
        print("\nWARNING: exact texts occur in both train and test.")
        print("Do not automatically remove them yet.")
        print("Inspect and document before modifying the original split.")


def validate(dataset):
    assert set(dataset.keys()) == {"train", "test"}

    assert len(dataset["train"]) == 7500, (
        f"Expected 7500 train examples, "
        f"found {len(dataset['train'])}"
    )

    assert len(dataset["test"]) == 2500, (
        f"Expected 2500 test examples, "
        f"found {len(dataset['test'])}"
    )

    for split_name, split in dataset.items():

        assert split.column_names == ["text", "label"]

        # No empty texts
        empty = [
            i for i, text in enumerate(split["text"])
            if not isinstance(text, str) or not text.strip()
        ]

        assert not empty, (
            f"{split_name} contains empty texts at: {empty[:10]}"
        )

        # Only expected labels
        labels = set(split["label"])
        assert labels.issubset({0, 1}), (
            f"Unexpected labels in {split_name}: {labels}"
        )

        counts = Counter(split["label"])

        print(
            f"{split_name}: "
            f"{len(split):,} examples | "
            f"NOT={counts[0]:,} | "
            f"OFF={counts[1]:,}"
        )


def prepare():
    print(f"Loading original SOLD dataset: {SOURCE_REPO}")

    raw = load_dataset(SOURCE_REPO)

    print("Available splits:", list(raw.keys()))
    print("Original columns:")
    for split in raw:
        print(f"  {split}: {raw[split].column_names}")

    prepared = DatasetDict()

    for split_name in ["train", "test"]:
        if split_name not in raw:
            raise ValueError(
                f"Missing expected split: {split_name}"
            )

        prepared[split_name] = raw[split_name].map(
            convert_example,
            remove_columns=raw[split_name].column_names,
            desc=f"Preparing {split_name}",
        )

    return prepared


def main():
    dataset = prepare()

    print("\nValidating dataset")
    print("==================")
    validate(dataset)

    check_duplicates(dataset)

    print("\nExample")
    print("=======")
    print(ascii(dataset["train"][0]))

    print(f"\nUploading to {TARGET_REPO}")

    dataset.push_to_hub(TARGET_REPO)

    api = HfApi()
    info = api.dataset_info(TARGET_REPO)

    print("\nDone.")
    print(f"Dataset:  {TARGET_REPO}")
    print(f"Revision: {info.sha}")


if __name__ == "__main__":
    main()