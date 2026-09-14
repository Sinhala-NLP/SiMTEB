#!/usr/bin/env python

import argparse

import mteb

from src.simteb.benchmark import get_benchmark_tasks
from src.simteb.registry import get_tasks


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an embedding model on Sinhala-MTEB."
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
        help="Optional task names. If omitted, run the full benchmark.",
    )

    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Optional task categories.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    if args.tasks and args.categories:
        raise ValueError(
            "Use either --tasks or --categories, not both."
        )

    print(f"Loading model: {args.model}")

    model = mteb.get_model(args.model)

    if args.tasks:
        tasks = get_tasks(names=args.tasks)

    elif args.categories:
        tasks = get_tasks(categories=args.categories)

    else:
        tasks = get_benchmark_tasks()

    print("\nTasks:")
    for task in tasks:
        print(f"  - {task.metadata.name}")

    results = mteb.evaluate(
        model,
        tasks=tasks,
        encode_kwargs={
            "batch_size": args.batch_size,
        },
    )

    print("\nResults")

    for result in results:
        print(
            f"{result.task_name}: "
            f"{result.get_score():.4f}"
        )


if __name__ == "__main__":
    main()