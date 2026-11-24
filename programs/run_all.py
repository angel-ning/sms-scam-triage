"""
run_all.py - run every experiment step in order.

  1. prepare_data.py      download dataset -> data/*.csv
  2. predict_base.py      compare 3 pretrained classifiers (Pipeline 1)
  3. finetune.py          fine-tune the chosen classifier
  4. predict_finetuned.py pretrained vs fine-tuned on the same test set
  5. zeroshot_select.py   compare 3 zero-shot taggers (Pipeline 2)

Uploading to the Hugging Face Hub is a separate step (push_to_hub.py) because
it needs your login.

Usage
  python run_all.py           # full run - use the GPU machine
  python run_all.py --smoke   # 64-message run of every step, checks the code only
"""

import argparse

import finetune
import predict_base
import predict_finetuned
import prepare_data
import zeroshot_select
from utils import Timer


def main(smoke: bool = False) -> None:
    steps = [
        ("Prepare data", lambda: prepare_data.main()),
        ("Pipeline 1: pretrained comparison", lambda: predict_base.main(smoke)),
        ("Pipeline 1: fine-tuning", lambda: finetune.main(smoke)),
        ("Pipeline 1: pretrained vs fine-tuned", lambda: predict_finetuned.main(smoke)),
        ("Pipeline 2: zero-shot comparison", lambda: zeroshot_select.main(smoke)),
    ]
    with Timer() as total:
        for number, (title, step) in enumerate(steps, start=1):
            print(f"\n{'=' * 70}\nStep {number}/{len(steps)}: {title}\n{'=' * 70}")
            step()
    mode = "smoke test" if smoke else "full run"
    print(f"\nAll steps finished ({mode}) in {total.seconds / 60:.1f} minutes.")


if __name__ == "__main__":       # required on Windows for multiprocessing-safe start-up
    parser = argparse.ArgumentParser(description="Run every experiment step")
    parser.add_argument("--smoke", action="store_true", help="tiny sample, quick code check")
    main(parser.parse_args().smoke)
