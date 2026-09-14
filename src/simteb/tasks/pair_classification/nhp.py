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

            # IMPORTANT:
            # Replace this with the SHA printed by
            # prepare_nhp.py after upload.
            "revision": "REPLACE_WITH_HF_COMMIT_SHA",
        },

        type="PairClassification",

        # Paragraph/article -> sentence/headline
        category="p2s",

        modalities=[
            "text",
        ],

        eval_splits=[
            "test",
        ],

        eval_langs=[
            "sin-Sinh",
        ],

        # Standard MTEB PairClassification metric
        main_score="max_ap",

        domains=[
            "News",
        ],

        task_subtypes=[
            "Semantic Similarity",
        ],

        annotations_creators="derived",

        # Fill these with the appropriate values
        # once we finalise all benchmark metadata.
        date=None,
        license=None,
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )

    # These are already MTEB defaults, but I prefer
    # making them explicit in SiMTEB.
    input1_column_name = "sentence1"
    input2_column_name = "sentence2"
    label_column_name = "labels"

    # The relationship is asymmetric:
    #
    # sentence1 = full news article
    # sentence2 = candidate headline
    #
    # This allows E5-style models to use passage/query
    # behaviour where supported.
    input1_prompt_type = "document"
    input2_prompt_type = "query"