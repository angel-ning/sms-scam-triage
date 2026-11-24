"""
predict_base.py - Step 2: model selection for Pipeline 1 (scam vs safe).

Loops over the three pretrained Hugging Face SMS classifiers in
config.P1_CANDIDATES (plus the outside benchmark), scores each one on the
same held-out test set, and records accuracy, scam recall, false-positive
rate, runtime and model size.

Output
  results/p1_pretrained_comparison.csv
  results/predictions/p1_<model>.csv     (every prediction, for error analysis)

Usage
  python predict_base.py            # full held-out test set
  python predict_base.py --smoke    # 64 messages, quick code check
"""

import argparse

import config
import utils
from classify import evaluate_classifier

TABLE_COLUMNS = ["role", "model", "accuracy", "recall_scam", "false_positive_rate",
                 "f1_scam", "runtime_seconds", "ms_per_message", "parameters_millions"]


def safe_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def main(smoke: bool = False) -> None:
    utils.set_seed()
    device = utils.get_device()
    out_dir = utils.results_dir(smoke)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    test = utils.load_split("test", limit=config.SMOKE_SAMPLE_SIZE if smoke else None)
    print(f"Pipeline 1 model selection on {len(test)} held-out messages ({device})")

    # Systematic model selection: one loop over every model to compare.
    to_compare = [(m, labels, "candidate") for m, labels in config.P1_CANDIDATES.items()]
    to_compare += [(m, labels, "benchmark") for m, labels in config.P1_BENCHMARK.items()]

    rows = []
    for model_id, label_map, role in to_compare:
        print(f"\n> {role}: {model_id}")
        try:
            result, predictions = evaluate_classifier(model_id, label_map, test, role, device)
        except Exception as err:          # one broken model must not stop the comparison
            print(f"  ! skipped {model_id}: {err}")
            continue
        rows.append(result)
        predictions.to_csv(pred_dir / f"p1_{safe_name(model_id)}.csv",
                           index=False, encoding="utf-8")
        m = result.metrics
        print(f"  accuracy={m['accuracy']:.3f}  recall_scam={m['recall_scam']:.3f}  "
              f"FPR={m['false_positive_rate']:.3f}  runtime={result.runtime_seconds:.1f}s")

    if not rows:
        raise SystemExit("No model could be evaluated - check the internet connection.")

    table = utils.save_results(rows, out_dir / "p1_pretrained_comparison.csv")
    print()
    utils.print_table(table, TABLE_COLUMNS)
    print(f"\nModel chosen for fine-tuning (config.P1_CHOSEN_BASE): {config.P1_CHOSEN_BASE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare pretrained Pipeline 1 models")
    parser.add_argument("--smoke", action="store_true", help="tiny sample, quick code check")
    main(parser.parse_args().smoke)
