#!/usr/bin/env python3

import argparse
from collections import Counter

from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi


SOURCE_REPO = "sinhala-nlp/sinhala-headline-prediction"
TARGET_REPO = "sinhala-nlp/SiMTEB-NHP"


def convert_example(example):
    news_content = example["news_content"]
    headline = example["headline"]
    label = example["is_headline"]

    if news_content is None:
        news_content = ""

    if headline is None:
        headline = ""

    news_content = str(news_content).strip()
    headline = str(headline).strip()

    try:
        label = int(label)
    except (TypeError, ValueError):
        raise ValueError(
            f"Invalid is_headline label: {label!r}"
        )

    if label not in {0, 1}:
        raise ValueError(
            f"Expected binary label 0/1, found: {label!r}"
        )

    return {
        # Standard MTEB PairClassification column names
        "sentence1": news_content,
        "sentence2": headline,
        "labels": label,
    }


def prepare_dataset():
    print(
        f"Loading source dataset: {SOURCE_REPO}"
    )

    raw = load_dataset(SOURCE_REPO)

    print(
        "Available splits:",
        list(raw.keys()),
    )

    print("\nOriginal columns:")
    for split in raw:
        print(
            f"  {split}: "
            f"{raw[split].column_names}"
        )

    prepared = DatasetDict()

    for split_name, split in raw.items():

        print(
            f"\nPreparing {split_name}..."
        )

        prepared[split_name] = split.map(
            convert_example,
            remove_columns=split.column_names,
            desc=f"Preparing {split_name}",
        )

    return prepared


def validate(dataset):
    print("\nValidation")
    print("==========")

    expected_columns = {
        "sentence1",
        "sentence2",
        "labels",
    }

    for split_name, split in dataset.items():

        print(
            f"\nSplit: {split_name}"
        )

        actual_columns = set(
            split.column_names
        )

        if actual_columns != expected_columns:
            raise ValueError(
                f"Unexpected columns in {split_name}: "
                f"{actual_columns}"
            )

        # -------------------------------------------------
        # Check empty documents/headlines
        # -------------------------------------------------

        empty_sentence1 = [
            i
            for i, text in enumerate(
                split["sentence1"]
            )
            if not isinstance(text, str)
            or not text.strip()
        ]

        empty_sentence2 = [
            i
            for i, text in enumerate(
                split["sentence2"]
            )
            if not isinstance(text, str)
            or not text.strip()
        ]

        if empty_sentence1:
            raise ValueError(
                f"{split_name}: "
                f"{len(empty_sentence1)} empty articles."
            )

        if empty_sentence2:
            raise ValueError(
                f"{split_name}: "
                f"{len(empty_sentence2)} empty headlines."
            )

        # -------------------------------------------------
        # Check labels
        # -------------------------------------------------

        labels = split["labels"]

        unique_labels = set(labels)

        if not unique_labels.issubset(
            {0, 1}
        ):
            raise ValueError(
                f"{split_name}: "
                f"unexpected labels: "
                f"{unique_labels}"
            )

        counts = Counter(labels)

        print(
            f"Examples: {len(split):,}"
        )

        print(
            f"Negative (0): {counts[0]:,}"
        )

        print(
            f"Positive (1): {counts[1]:,}"
        )

        # -------------------------------------------------
        # Exact pair duplicate audit
        #
        # Repeated articles are OK because the same article
        # can intentionally be paired with different
        # candidate headlines.
        # -------------------------------------------------

        pairs = list(
            zip(
                split["sentence1"],
                split["sentence2"],
            )
        )

        unique_pairs = set(pairs)

        exact_duplicates = (
            len(pairs)
            - len(unique_pairs)
        )

        print(
            f"Exact pair duplicates: "
            f"{exact_duplicates:,}"
        )


def audit_cross_split_overlap(dataset):
    if (
        "train" not in dataset
        or "test" not in dataset
    ):
        return

    train_pairs = set(
        zip(
            dataset["train"]["sentence1"],
            dataset["train"]["sentence2"],
        )
    )

    test_pairs = set(
        zip(
            dataset["test"]["sentence1"],
            dataset["test"]["sentence2"],
        )
    )

    pair_overlap = (
        train_pairs.intersection(
            test_pairs
        )
    )

    print("\nCross-split audit")
    print("=================")

    print(
        "Exact train/test pair overlap: "
        f"{len(pair_overlap):,}"
    )

    # -------------------------------------------------
    # Also check whether the same article occurs in
    # both train and test.
    #
    # Do NOT automatically remove anything.
    # We only report this for now.
    # -------------------------------------------------

    train_articles = set(
        dataset["train"]["sentence1"]
    )

    test_articles = set(
        dataset["test"]["sentence1"]
    )

    article_overlap = (
        train_articles.intersection(
            test_articles
        )
    )

    print(
        "Article overlap between train/test: "
        f"{len(article_overlap):,}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        default=TARGET_REPO,
        help=(
            "Target Hugging Face dataset "
            "repository."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Prepare and validate the dataset "
            "without uploading."
        ),
    )

    args = parser.parse_args()

    dataset = prepare_dataset()

    validate(dataset)

    audit_cross_split_overlap(
        dataset
    )

    print("\nExample")
    print("=======")

    # ascii() avoids the Unicode terminal problem
    # encountered previously on Wayland.
    print(
        ascii(dataset["train"][0])
    )

    if args.dry_run:
        print(
            "\nDry run finished. "
            "Nothing uploaded."
        )
        return

    print(
        f"\nUploading to "
        f"{args.repo_id}"
    )

    dataset.push_to_hub(
        args.repo_id
    )

    api = HfApi()

    info = api.dataset_info(
        args.repo_id
    )

    print("\nUpload complete")
    print("===============")

    print(
        f"Repository: "
        f"{args.repo_id}"
    )

    print(
        f"Revision:   "
        f"{info.sha}"
    )


if __name__ == "__main__":
    main()