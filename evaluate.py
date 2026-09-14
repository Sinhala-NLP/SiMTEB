#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import mteb

from simteb.benchmark import get_benchmark_tasks
from simteb.registry import get_tasks


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an embedding model on SiMTEB."
    )

    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Hugging Face model name, MTEB model name, "
            "or local model path."
        ),
    )

    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help=(
            "Optional individual task names. "
            "If omitted, the full SiMTEB benchmark is evaluated."
        ),
    )

    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help=(
            "Optional task categories. "
            "For example: Classification Clustering"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size.",
    )

    parser.add_argument(
        "--cache-folder",
        default=None,
        help=(
            "Directory used by the MTEB result cache. "
            "If omitted, MTEB_CACHE is used when defined; "
            "otherwise ~/.cache/mteb is used."
        ),
    )

    return parser.parse_args()


def get_cache_folder(args):
    """
    Determine where MTEB should store its result files.

    Priority:
        1. --cache-folder
        2. MTEB_CACHE environment variable
        3. ~/.cache/mteb
    """

    if args.cache_folder:
        cache_folder = Path(args.cache_folder)

    elif os.environ.get("MTEB_CACHE"):
        cache_folder = Path(os.environ["MTEB_CACHE"])

    else:
        cache_folder = Path.home() / ".cache" / "mteb"

    cache_folder = cache_folder.expanduser().resolve()

    cache_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    return cache_folder


def select_tasks(args):
    if args.tasks and args.categories:
        raise ValueError(
            "Specify either --tasks or --categories, not both."
        )

    if args.tasks:
        return get_tasks(
            names=args.tasks
        )

    if args.categories:
        return get_tasks(
            categories=args.categories
        )

    return get_benchmark_tasks()


def main():
    args = parse_args()

    cache_folder = get_cache_folder(args)

    print("=" * 70)
    print("SiMTEB")
    print("=" * 70)
    print(f"Model:       {args.model}")
    print(f"Batch size:  {args.batch_size}")
    print(f"MTEB cache:  {cache_folder}")
    print(f"HF_HOME:     {os.environ.get('HF_HOME', 'not set')}")
    print(
        f"HF datasets: "
        f"{os.environ.get('HF_DATASETS_CACHE', 'not set')}"
    )
    print(
        f"XDG cache:   "
        f"{os.environ.get('XDG_CACHE_HOME', 'not set')}"
    )
    print("=" * 70)

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    print(f"\nLoading model: {args.model}")

    model = mteb.get_model(
        args.model
    )

    # ---------------------------------------------------------
    # Select tasks
    # ---------------------------------------------------------

    tasks = select_tasks(args)

    print("\nTasks:")

    for task in tasks:
        print(
            f"  - {task.metadata.name}"
        )

    # ---------------------------------------------------------
    # Explicit MTEB cache
    # ---------------------------------------------------------

    cache = mteb.ResultCache(
        cache_path=cache_folder
    )

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    print("\nStarting evaluation...\n")

    results = mteb.evaluate(
        model,
        tasks=tasks,
        cache=cache,
        encode_kwargs={
            "batch_size": args.batch_size,
        },
    )

    # ---------------------------------------------------------
    # Print results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("SiMTEB Results")
    print("=" * 70)

    for result in results:
        try:
            score = result.get_score()

            print(
                f"{result.task_name}: "
                f"{score:.4f}"
            )

        except Exception:
            # Do not crash simply because the output interface
            # changed between MTEB versions.
            print(result)

    print("=" * 70)
    print(
        f"Full MTEB result files are stored in:\n"
        f"{cache_folder}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()