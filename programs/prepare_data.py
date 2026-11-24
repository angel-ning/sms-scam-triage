"""
prepare_data.py - Step 1: download the dataset and save it as CSV files in data/.

Output
  data/train.csv, data/validation.csv, data/test.csv   (Pipeline 1: scam vs safe)
  data/p2_test.csv                                      (Pipeline 2: scam-type tagging)
  results/dataset_summary.json                          (counts quoted in the report)

The CSVs are committed to the repository, so later steps (and the graders)
run on exactly the same data even without internet access.
"""

import argparse

import pandas as pd

import config
import utils


def download_splits() -> dict[str, pd.DataFrame]:
    """Download every split from the Hugging Face Hub."""
    from datasets import load_dataset
    dataset = load_dataset(config.DATASET_ID)
    return {split: dataset[split].to_pandas() for split in config.SPLITS}


def build_type_split(test: pd.DataFrame) -> pd.DataFrame:
    """Keep real scam messages whose scam type maps to one of our queues."""
    scams = test[test[config.LABEL_COL] == config.LABEL_SCAM].copy()
    scams["queue"] = scams[config.TYPE_COL].map(config.P2_TYPE_MAP)
    return scams.dropna(subset=["queue"])[[config.TEXT_COL, config.TYPE_COL, "queue"]]


def summarise(splits: dict[str, pd.DataFrame], type_split: pd.DataFrame) -> dict:
    summary = {"dataset": config.DATASET_ID, "splits": {}}
    for name, df in splits.items():
        summary["splits"][name] = {
            "rows": len(df),
            "labels": df[config.LABEL_COL].value_counts().to_dict(),
            "sources": df[config.SOURCE_COL].value_counts().to_dict(),
        }
    summary["p2_test"] = {"rows": len(type_split),
                          "queues": type_split["queue"].value_counts().to_dict()}
    return summary


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        splits = download_splits()
        for name, df in splits.items():
            df.to_csv(utils.split_path(name), index=False, encoding="utf-8")
        print(f"Downloaded {config.DATASET_ID} and saved CSVs to {config.DATA_DIR}")
    except Exception as err:                      # no internet / Hub outage
        print(f"Download failed ({err}). Falling back to the CSVs already in data/.")
        splits = {name: utils.load_split(name) for name in config.SPLITS}

    type_split = build_type_split(splits["test"])
    type_split.to_csv(utils.split_path("p2_test"), index=False, encoding="utf-8")

    summary = summarise(splits, type_split)
    utils.save_json(summary, utils.results_dir(smoke=False) / "dataset_summary.json")

    print("\nDataset summary")
    for name, info in summary["splits"].items():
        print(f"  {name:<11} {info['rows']:>6} rows  labels={info['labels']}")
    print(f"  p2_test     {summary['p2_test']['rows']:>6} rows  queues={summary['p2_test']['queues']}")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
