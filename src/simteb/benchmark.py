from src.simteb.registry import get_tasks


SIMTEB_TASKS = [
    "SOLDClassification",
    "SinhalaHeadlinePrediction",
]


def get_benchmark_tasks():
    return get_tasks(
        names=SIMTEB_TASKS
    )