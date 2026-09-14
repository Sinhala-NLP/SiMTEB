from simteb.tasks.classification import (
    SOLDClassification,
)

from simteb.tasks.pair_classification import (
    SinhalaHeadlinePrediction,
)


TASK_REGISTRY = {
    "SOLDClassification":
        SOLDClassification,

    "SinhalaHeadlinePrediction":
        SinhalaHeadlinePrediction,
}


CATEGORY_REGISTRY = {
    "Classification": [
        "SOLDClassification",
    ],

    "PairClassification": [
        "SinhalaHeadlinePrediction",
    ],
}


def get_task(name):
    if name not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task: {name}. "
            f"Available tasks: "
            f"{list(TASK_REGISTRY)}"
        )

    return TASK_REGISTRY[name]()


def get_tasks(
    names=None,
    categories=None,
):
    if names is not None:
        return [
            get_task(name)
            for name in names
        ]

    if categories is not None:
        selected_names = []

        for category in categories:

            if category not in CATEGORY_REGISTRY:
                raise KeyError(
                    f"Unknown category: "
                    f"{category}"
                )

            selected_names.extend(
                CATEGORY_REGISTRY[
                    category
                ]
            )

        # Preserve order while removing duplicates.
        selected_names = list(
            dict.fromkeys(
                selected_names
            )
        )

        return [
            get_task(name)
            for name in selected_names
        ]

    return [
        task_class()
        for task_class
        in TASK_REGISTRY.values()
    ]