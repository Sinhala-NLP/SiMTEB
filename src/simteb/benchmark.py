from src.simteb.registry import get_tasks


SINHALA_MTEB_TASKS = [
    "SOLDClassification",

    # Later:
    # "SinhalaSentimentClassification",
    # "SinhalaEmotionClassification",
    #
    # "NSinaHeadlineClustering",
    # "NSinaArticleClustering",
    #
    # "MUSTSSTS",
    #
    # "FloresSinhalaEnglishBitextMining",
    # "TamSiParaBitextMining",
    # "PaliSinhalaBitextMining",
    #
    # "SinhalaEnglishQEDA",
    # "SinhalaEnglishQEHTER",
]


def get_benchmark_tasks():
    return get_tasks(names=SINHALA_MTEB_TASKS)