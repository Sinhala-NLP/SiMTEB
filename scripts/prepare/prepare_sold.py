#!/usr/bin/env python

import argparse

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi


SOURCE_REPO = "sinhala-nlp/SOLD"

LABEL_MAP = {
    "NOT": 0,
    "OFF": 1,
}


def convert_example(example):
    label = example["label"]

    if label not in LABEL_MAP:
        raise ValueError(f"Unexpected SOLD label: {label}")

    return {
        "text": example["text"],
        "label": LABEL_MAP[label],
    }


def prepare_sold():
    print(f"Loading SOLD from {SOURCE_REPO}")

    dataset = load_dataset(SOURCE_REPO)

    output = DatasetDict()

    for split in ["train", "test"]:
        if split not in dataset:
            raise ValueError(
                f"Expected split '{split}' not found. "
                f"Available splits: {list(dataset.keys())}"
            )

        converted = dataset[split].map(
            convert_example,
            remove_columns=dataset[split].column_names,
        )

        output[split] = converted

    return output


def validate(dataset):
    expected_splits = {"train", "test"}

    if set(dataset.keys()) != expected_splits:
        raise ValueError(
            f"Unexpected splits: {list(dataset.keys())}"
        )

    # SOLD should contain 7,500 train and 2,500 test examples.
    assert len(dataset["train"]) == 7500
    assert len(dataset["test"]) == 2500

    for split in dataset:
        assert dataset[split].column_names == ["text", "label"]

        labels = set(dataset[split]["label"])
        assert labels.issubset({0, 1})

        assert all(
            isinstance(text, str) and text.strip()
            for text in dataset[split]["text"]
        )

    print("Validation passed.")

    for split in dataset:
        labels = dataset[split]["label"]

        print(
            f"{split}: "
            f"{len(dataset[split])} examples | "
            f"NOT={labels.count(0)} | "
            f"OFF={labels.count(1)}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        default="Sinhala-NLP/Sinhala-MTEB-SOLD",
        help="Target Hugging Face dataset repository.",
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the target repository as private.",
    )

    args = parser.parse_args()

    dataset = prepare_sold()

    validate(dataset)

    print("\nExample:")
    print(dataset["train"][0])

    print(f"\nUploading to {args.repo_id}")

    dataset.push_to_hub(
        args.repo_id,
        private=args.private,
    )

    # Important: print the immutable revision so that the benchmark
    # can later pin the exact dataset version.
    api = HfApi()
    info = api.dataset_info(args.repo_id)

    print("\nUpload complete.")
    print(f"Repository: {args.repo_id}")
    print(f"Revision SHA: {info.sha}")


if __name__ == "__main__":
    main()