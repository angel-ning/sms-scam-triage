"""
predict_finetuned.py - Step 4: pretrained vs fine-tuned on the SAME held-out test set.

Output
  results/p1_pretrained_vs_finetuned.csv
  results/p1_finetuned_errors.csv          (every misclassified message, for the report)

Usage
  python predict_finetuned.py            # needs models/finetuned-scam-classifier
  python predict_finetuned.py --smoke    # uses the smoke-test model
"""

import argparse

import config
import utils
from classify import evaluate_classifier
from predict_base import TABLE_COLUMNS


def main(smoke: bool = False) -> None:
    utils.set_seed()
    device = utils.get_device()
    out_dir = utils.results_dir(smoke)
    model_dir = utils.finetuned_dir(smoke)
    if not model_dir.exists():
        raise SystemExit(f"{model_dir} not found - run finetune.py first.")

    test = utils.load_split("test", limit=config.SMOKE_SAMPLE_SIZE if smoke else None)
    print(f"Pretrained vs fine-tuned on {len(test)} held-out messages ({device})")

    versions = [
        (config.P1_CHOSEN_BASE, config.P1_CANDIDATES[config.P1_CHOSEN_BASE], "pretrained"),
        (str(model_dir), {config.LABEL_SAFE: config.LABEL_SAFE,
                          config.LABEL_SCAM: config.LABEL_SCAM}, "fine-tuned"),
    ]
    rows = []
    for model_id, label_map, role in versions:
        print(f"\n> {role}: {model_id}")
        result, predictions = evaluate_classifier(model_id, label_map, test, role, device)
        if role == "fine-tuned":
            result.model = f"{config.P1_CHOSEN_BASE} (fine-tuned on {config.DATASET_ID})"
            errors = predictions[predictions["predicted"] != predictions[config.LABEL_COL]]
            errors.to_csv(out_dir / "p1_finetuned_errors.csv", index=False, encoding="utf-8")
            print(f"  {len(errors)} misclassified messages saved for error analysis")
        rows.append(result)

    table = utils.save_results(rows, out_dir / "p1_pretrained_vs_finetuned.csv")
    print()
    utils.print_table(table, TABLE_COLUMNS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare pretrained and fine-tuned models")
    parser.add_argument("--smoke", action="store_true", help="tiny sample, quick code check")
    main(parser.parse_args().smoke)
