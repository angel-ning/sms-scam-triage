"""
analyze_live_test.py - Step 7: score the live-app tests against the success criteria.

Input
  data/live_app_test.csv       50 realistic US messages written by the group (not taken
                               from the training data), with the true label, the expected
                               scam type and a category:
                                 scam_typical  - brand / link / urgency scams
                                 scam_casual   - personal-tone scams without links (hard cases)
                                 safe_service  - real-looking service alerts (bills, codes, deliveries)
                                 safe_personal - everyday personal chats
  results/live_app_results.csv the "Download results (CSV)" file from the deployed app's
                               Batch tab after uploading data/live_app_test.csv

Output
  results/live_app_test_table.csv    one row per message: truth, app output, correct or not
  results/live_app_summary.json      accuracy and the success-criteria checks

Usage
  python analyze_live_test.py           # score the file downloaded from the live app
  python analyze_live_test.py --local   # run the app's own triage code locally on CPU
                                        # (preview / reproduction without the website)
"""

import argparse
import sys

import pandas as pd

import config
import utils

LIVE_TEST_FILE = config.DATA_DIR / "live_app_test.csv"
APP_RESULTS_FILE = config.RESULTS_DIR / "live_app_results.csv"
LOCAL_RESULTS_FILE = config.RESULTS_DIR / "live_app_results_local.csv"

ACCURACY_TARGET = 0.95            # success criterion 1 (also checked on live messages)
SERVICE_FP_LIMIT = 0.10           # success criterion 2: <= 10% of service alerts flagged


def run_app_locally() -> pd.DataFrame:
    """Produce the same table the app's Batch tab would download."""
    sys.path.insert(0, str(config.ROOT_DIR / "app"))
    import app                                   # app.main() does not run on import
    from transformers import pipeline

    live = pd.read_csv(LIVE_TEST_FILE, encoding="utf-8")
    classifier = pipeline("text-classification", model=config.HF_MODEL_REPO, device="cpu")
    tagger = pipeline("zero-shot-classification", model=config.P2_CHOSEN_MODEL, device="cpu")
    table = app.triage(app.clean_inputs(live["text"].tolist()), classifier, tagger,
                       app.DEFAULT_REVIEW_THRESHOLD)
    table.to_csv(LOCAL_RESULTS_FILE, index=False, encoding="utf-8")
    print(f"Saved local app output -> {LOCAL_RESULTS_FILE}")
    return table


def load_app_results() -> pd.DataFrame:
    if not APP_RESULTS_FILE.exists():
        raise SystemExit(f"{APP_RESULTS_FILE} not found. Upload data/live_app_test.csv in the "
                         "app's Batch tab, click 'Download results (CSV)' and save it there.")
    return pd.read_csv(APP_RESULTS_FILE, encoding="utf-8")


def merge(live: pd.DataFrame, output: pd.DataFrame) -> pd.DataFrame:
    """Join the app output (sorted by priority) back to the test set by message text."""
    output = output.rename(columns={"Message": "text"}).drop(columns=["True label"], errors="ignore")
    merged = live.merge(output, on="text", how="left", validate="one_to_one")
    missing = merged["Verdict"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} test message(s) are missing from the app output - "
                         "was the right file uploaded?")
    merged["correct"] = merged["Verdict"] == merged["label"]
    is_scam = merged["label"] == config.LABEL_SCAM
    merged["type_correct"] = (merged["Scam type"] == merged["expected_type"]).where(
        is_scam & (merged["Verdict"] == config.LABEL_SCAM))
    return merged.sort_values("id")


def summarise(table: pd.DataFrame) -> dict:
    overall = utils.binary_metrics(table["label"].tolist(), table["Verdict"].tolist())
    by_category = {
        name: {"messages": len(group), "correct": int(group["correct"].sum()),
               "accuracy": round(group["correct"].mean(), 4)}
        for name, group in table.groupby("category")
    }
    service = table[table["category"] == "safe_service"]
    service_fp_rate = float((service["Verdict"] == config.LABEL_SCAM).mean())
    typed = table["type_correct"].dropna()

    return {
        "messages": len(table),
        "accuracy": round(overall["accuracy"], 4),
        "recall_scam": round(overall["recall_scam"], 4),
        "false_positive_rate": round(overall["false_positive_rate"], 4),
        "confusion": {k: overall[k] for k in ("tp", "fp", "tn", "fn")},
        "by_category": by_category,
        "sent_to_review": int((table["Priority"].str.contains("Review")).sum()),
        "scam_type_accuracy": round(float(typed.mean()), 4) if len(typed) else None,
        "scam_type_scored_on": int(len(typed)),
        "criterion_1_live_accuracy_met": bool(overall["accuracy"] >= ACCURACY_TARGET),
        "criterion_2_service_false_positive_rate": round(service_fp_rate, 4),
        "criterion_2_met": bool(service_fp_rate <= SERVICE_FP_LIMIT),
    }


def main(local: bool = False) -> None:
    live = pd.read_csv(LIVE_TEST_FILE, encoding="utf-8", keep_default_na=False)
    output = run_app_locally() if local else load_app_results()
    table = merge(live, output)
    summary = summarise(table)

    suffix = "_local" if local else ""
    columns = ["id", "category", "text", "label", "Verdict", "Confidence", "correct",
               "expected_type", "Scam type", "Type confidence", "type_correct", "Priority"]
    table[columns].to_csv(config.RESULTS_DIR / f"live_app_test_table{suffix}.csv",
                          index=False, encoding="utf-8")
    utils.save_json(summary, config.RESULTS_DIR / f"live_app_summary{suffix}.json")

    print(f"\nLive-app test: {summary['accuracy']:.1%} accuracy on {summary['messages']} messages "
          f"(scam recall {summary['recall_scam']:.1%}, false-positive rate "
          f"{summary['false_positive_rate']:.1%}, {summary['sent_to_review']} sent to review)")
    for name, info in summary["by_category"].items():
        print(f"  {name:<14} {info['correct']:>2}/{info['messages']:<2}  {info['accuracy']:.0%}")
    print(f"Scam-type accuracy: {summary['scam_type_accuracy']:.0%} "
          f"(on {summary['scam_type_scored_on']} correctly flagged scams)")
    print(f"Criterion 2 - service alerts wrongly flagged: "
          f"{summary['criterion_2_service_false_positive_rate']:.0%} "
          f"(limit {SERVICE_FP_LIMIT:.0%}) -> {'met' if summary['criterion_2_met'] else 'NOT met'}")

    wrong = table[~table["correct"]]
    if len(wrong):
        print("\nMisclassified:")
        for row in wrong.itertuples():
            print(f"  #{row.id} [{row.category}] true={row.label} app={row.Verdict} "
                  f"({row.Confidence:.0%}): {row.text[:80]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score the live-app tests")
    parser.add_argument("--local", action="store_true",
                        help="run the app's triage code locally instead of reading the download")
    main(parser.parse_args().local)
