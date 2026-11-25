"""
classify.py - the shared "run a text-classification pipeline on a test set"
routine used by predict_base.py and predict_finetuned.py (so neither script
repeats the evaluation code).
"""

import pandas as pd
from transformers import pipeline

import config
import utils


def load_classifier(model_id: str, device: str):
    """Build a Hugging Face text-classification pipeline.

    Some operators are not implemented on Apple's MPS backend; if loading or a
    warm-up call fails there, fall back to the CPU instead of crashing.
    """
    try:
        clf = pipeline("text-classification", model=model_id, device=device)
        clf("warm-up message", truncation=True, max_length=config.MAX_TOKENS)
        return clf, device
    except Exception as err:
        if device == "cpu":
            raise
        print(f"  ! {model_id} failed on {device} ({err}); retrying on CPU")
        return pipeline("text-classification", model=model_id, device="cpu"), "cpu"


def evaluate_classifier(model_id: str, label_map: dict[str, str], test: pd.DataFrame,
                        role: str, device: str) -> tuple[utils.ModelResult, pd.DataFrame]:
    """Score one model on the held-out set; return its result row and its predictions."""
    clf, used_device = load_classifier(model_id, device)
    texts = test[config.TEXT_COL].tolist()

    # Untimed warm-up on one full batch: the first batch on a GPU pays one-off
    # start-up costs (kernel loading, memory allocation). Without this, the
    # first model in the loop looks slower than it really is.
    clf(texts[:config.INFER_BATCH_SIZE], batch_size=config.INFER_BATCH_SIZE,
        truncation=True, max_length=config.MAX_TOKENS)

    with utils.Timer() as timer:
        outputs = clf(texts, batch_size=config.INFER_BATCH_SIZE,
                      truncation=True, max_length=config.MAX_TOKENS)

    raw_labels = [o["label"] for o in outputs]
    unknown = set(raw_labels) - set(label_map)
    if unknown:
        raise ValueError(f"{model_id} produced labels {unknown} that are not in its label map")
    predicted = [label_map[label] for label in raw_labels]

    metrics = utils.binary_metrics(test[config.LABEL_COL].tolist(), predicted)
    result = utils.ModelResult(
        pipeline="P1 scam vs safe", role=role, model=model_id,
        n_samples=len(test), runtime_seconds=timer.seconds,
        ms_per_message=1000 * timer.seconds / len(test),
        parameters_millions=utils.count_parameters_millions(clf.model),
        device=utils.device_name(used_device), metrics=metrics,
    )

    predictions = test[[config.TEXT_COL, config.LABEL_COL]].copy()
    predictions["predicted"] = predicted
    predictions["score"] = [round(o["score"], 4) for o in outputs]
    return result, predictions
