"""
zeroshot_select.py - Step 5: model selection for Pipeline 2 (scam-type tagging).

Each scam message is routed to one analyst queue (banking, delivery,
government, telecom, personal) with a zero-shot-classification pipeline - no
fine-tuning. Three pretrained NLI models are compared in one loop on the
real scam messages of the held-out test split (data/p2_test.csv).

Output
  results/p2_zeroshot_comparison.csv
  results/predictions/p2_<model>.csv

Usage
  python zeroshot_select.py            # full set
  python zeroshot_select.py --smoke    # 64 messages, quick code check
"""

import argparse

from transformers import pipeline

import config
import utils
from predict_base import safe_name

TABLE_COLUMNS = ["model", "accuracy", "macro_f1", "runtime_seconds",
                 "ms_per_message", "parameters_millions"]

# Reverse look-up: description shown to the model -> queue name
DESCRIPTION_TO_QUEUE = {desc: queue for queue, desc in config.P2_LABEL_DESCRIPTIONS.items()}


def load_zero_shot(model_id: str, device: str):
    try:
        tagger = pipeline("zero-shot-classification", model=model_id, device=device)
        tagger("warm-up", candidate_labels=["a bank"], hypothesis_template=config.P2_HYPOTHESIS)
        return tagger, device
    except Exception as err:
        if device == "cpu":
            raise
        print(f"  ! {model_id} failed on {device} ({err}); retrying on CPU")
        return pipeline("zero-shot-classification", model=model_id, device="cpu"), "cpu"


def evaluate_tagger(model_id: str, data, device: str):
    tagger, used_device = load_zero_shot(model_id, device)
    texts = data[config.TEXT_COL].tolist()

    # Untimed warm-up on one full batch (same reason as in classify.py)
    tagger(texts[:config.INFER_BATCH_SIZE], candidate_labels=list(DESCRIPTION_TO_QUEUE),
           hypothesis_template=config.P2_HYPOTHESIS, batch_size=config.INFER_BATCH_SIZE)

    with utils.Timer() as timer:
        outputs = tagger(texts,
                         candidate_labels=list(DESCRIPTION_TO_QUEUE),
                         hypothesis_template=config.P2_HYPOTHESIS,
                         batch_size=config.INFER_BATCH_SIZE)
    predicted = [DESCRIPTION_TO_QUEUE[o["labels"][0]] for o in outputs]   # top-scoring label

    result = utils.ModelResult(
        pipeline="P2 scam-type tagging", role="candidate", model=model_id,
        n_samples=len(data), runtime_seconds=timer.seconds,
        ms_per_message=1000 * timer.seconds / len(data),
        parameters_millions=utils.count_parameters_millions(tagger.model),
        device=utils.device_name(used_device),
        metrics=utils.multiclass_metrics(data["queue"].tolist(), predicted),
    )
    predictions = data.copy()
    predictions["predicted"] = predicted
    predictions["score"] = [round(o["scores"][0], 4) for o in outputs]
    return result, predictions


def main(smoke: bool = False) -> None:
    utils.set_seed()
    device = utils.get_device()
    out_dir = utils.results_dir(smoke)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    data = utils.load_split("p2_test", limit=config.SMOKE_SAMPLE_SIZE if smoke else None)
    print(f"Pipeline 2 model selection on {len(data)} real scam messages ({device})")
    print(f"Queue sizes: {data['queue'].value_counts().to_dict()}")

    rows = []
    for model_id in config.P2_CANDIDATES:          # systematic loop over candidates
        print(f"\n> candidate: {model_id}")
        try:
            result, predictions = evaluate_tagger(model_id, data, device)
        except Exception as err:
            print(f"  ! skipped {model_id}: {err}")
            continue
        rows.append(result)
        predictions.to_csv(pred_dir / f"p2_{safe_name(model_id)}.csv", index=False, encoding="utf-8")
        print(f"  accuracy={result.metrics['accuracy']:.3f}  macro_f1={result.metrics['macro_f1']:.3f}  "
              f"runtime={result.runtime_seconds:.1f}s")

    if not rows:
        raise SystemExit("No zero-shot model could be evaluated.")
    table = utils.save_results(rows, out_dir / "p2_zeroshot_comparison.csv")
    print()
    utils.print_table(table, TABLE_COLUMNS)
    print(f"\nModel used in the app (config.P2_CHOSEN_MODEL): {config.P2_CHOSEN_MODEL}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare zero-shot models for Pipeline 2")
    parser.add_argument("--smoke", action="store_true", help="tiny sample, quick code check")
    main(parser.parse_args().smoke)
