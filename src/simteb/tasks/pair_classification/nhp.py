import mteb
from mteb.abstasks import AbsTaskPairClassification


class SinhalaHeadlinePrediction(
    AbsTaskPairClassification
):

    metadata = mteb.TaskMetadata(
        name="SinhalaHeadlinePrediction",

        description=(
            "Pair classification task for determining "
            "whether a candidate Sinhala news headline "
            "corresponds to a Sinhala news article."
        ),

        reference=(
            "https://huggingface.co/datasets/"
            "sinhala-nlp/sinhala-headline-prediction"
        ),

        dataset={
            "path": "sinhala-nlp/SiMTEB-NHP",

            # Replace with the commit SHA printed by
            # prepare_nhp.py.
            "revision": "15cbde77537a2af61ab1d5c9807d77371d7d7e2a",
        },

        type="PairClassification",

        # Current MTEB uses modality-based categories.
        # Both inputs are text.
        category="t2t",

        modalities=[
            "text",
        ],

        eval_splits=[
            "test",
        ],

        eval_langs=[
            "sin-Sinh",
        ],

        # Standard main score for MTEB pair classification.
        main_score="max_ap",

        date=(
            "2024-01-01",
            "2024-12-31",
        ),

        domains=[
            "News",
            "Written",
        ],

        license=None,

        annotations_creators="derived",

        dialect=[],

        sample_creation="found",

        bibtex_citation="",
    )

    input1_column_name = "sentence1"
    input2_column_name = "sentence2"
    label_column_name = "labels"

    # Full article is treated as the document/passage side.
    input1_prompt_type = "document"

    # Headline is treated as the query side.
    input2_prompt_type = "query"