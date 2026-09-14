import mteb
from mteb.abstasks import AbsTaskClassification


class SOLDClassification(AbsTaskClassification):
    metadata = mteb.TaskMetadata(
        name="SOLDClassification",
        description=(
            "Sinhala offensive language identification. "
            "Given a Sinhala social-media post, classify it as "
            "offensive or not offensive."
        ),
        type="Classification",
        main_score="f1",
        eval_langs=["sin-Sinh"],
        eval_splits=["test"],
        dataset={
            "path": "sinhala-nlp/SiMTEB-SOLD",

            # Replace this after running prepare_sold.py.
            "revision": "c42e5cbd050d78ef73f056cb9a2de13e06166269",
        },
        prompt=(
            "Classify the Sinhala social-media post as "
            "offensive or not offensive."
        ),
    )

    input_column_name = "text"
    label_column_name = "label"