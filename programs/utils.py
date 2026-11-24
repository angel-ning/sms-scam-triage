"""
utils.py - helpers shared by every program, so no code is copy-pasted.

Data-structure choices (required by the brief):
  * pandas DataFrame  - each split is a table of text + label columns; vectorised
                        filtering / sampling and one-line CSV export.
  * dict              - label translations and metric bundles: O(1) look-up by name.
  * dataclass         - ModelResult gives every experiment row the same fixed
                        fields, so rows from different models line up in one table.
  * list              - an ordered collection of ModelResult rows that becomes a
                        DataFrame / CSV at the end.
"""

import json
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Reproducibility and hardware
# ---------------------------------------------------------------------------
def set_seed(seed: int = config.SEED) -> None:
    """Fix every random source so a rerun gives the same numbers as the report."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Pick the fastest available device: NVIDIA GPU, then Apple GPU, then CPU."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_name(device: str) -> str:
    """Human-readable hardware name, stored next to every runtime figure."""
    import platform
    import torch
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return f"Apple GPU (MPS) on {platform.machine()}"
    return f"CPU ({platform.processor() or platform.machine()})"


# ---------------------------------------------------------------------------
# Output folders (real run vs smoke test)
# ---------------------------------------------------------------------------
def results_dir(smoke: bool) -> Path:
    path = config.RESULTS_DIR / config.SMOKE_SUBDIR if smoke else config.RESULTS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def finetuned_dir(smoke: bool) -> Path:
    base = config.MODELS_DIR / config.SMOKE_SUBDIR if smoke else config.MODELS_DIR
    return base / config.FINETUNED_DIRNAME


# ---------------------------------------------------------------------------
# Data loading and text cleaning
# ---------------------------------------------------------------------------
def split_path(split: str) -> Path:
    return config.DATA_DIR / f"{split}.csv"


def load_split(split: str, limit: int | None = None) -> pd.DataFrame:
    """Read one split from data/. `limit` draws a fixed random sample (smoke tests)."""
    path = split_path(split)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run prepare_data.py first.")
    df = pd.read_csv(path, encoding="utf-8")
    df[config.TEXT_COL] = df[config.TEXT_COL].astype(str)
    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=config.SEED).reset_index(drop=True)
    return df


# The dataset hides links, phone numbers, e-mails and long numbers behind
# placeholder tags. Live messages typed into the app must be masked the same
# way, otherwise the fine-tuned model sees text unlike anything it trained on.
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+"                                   # explicit links
    r"|\b(?:[a-z0-9-]+\.)+(?:com|net|org|info|xyz|top|us|co|io|me|ly|link|"
    r"site|online|shop|app|live|club|vip|cc)\b(?:/\S*)?",       # bare domains
    re.IGNORECASE,
)
_PHONE_RE = re.compile(r"(?<![\w<])\+?\d[\d\s().-]{6,}\d(?![\w>])")
_LONG_NUMBER_RE = re.compile(r"(?<![\w<])\d{4,}(?![\w>])")

MASK_RULES = (            # order matters: e-mails before URLs, phones before long numbers
    (_EMAIL_RE, "<EMAIL_ADDRESS>"),
    (_URL_RE, "<URL>"),
    (_PHONE_RE, "<PHONE_NUMBER>"),
    (_LONG_NUMBER_RE, "<NUMBER>"),
)


def mask_text(text: str) -> str:
    """Replace personal / variable details with the dataset's placeholder tags."""
    for pattern, tag in MASK_RULES:
        text = pattern.sub(tag, text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def binary_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Accuracy plus the error types that matter to a fraud team.

    * recall (scam)        - share of real scams we catch (missed scams reach customers)
    * false positive rate  - share of safe messages wrongly flagged (wastes analyst time)
    """
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 f1_score, precision_score, recall_score)
    labels = [config.LABEL_SAFE, config.LABEL_SCAM]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_scam": precision_score(y_true, y_pred, pos_label=config.LABEL_SCAM, zero_division=0),
        "recall_scam": recall_score(y_true, y_pred, pos_label=config.LABEL_SCAM, zero_division=0),
        "f1_scam": f1_score(y_true, y_pred, pos_label=config.LABEL_SCAM, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def multiclass_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


# ---------------------------------------------------------------------------
# Experiment records
# ---------------------------------------------------------------------------
@dataclass
class ModelResult:
    """One row of an experiment table (one model on one held-out set)."""
    pipeline: str
    role: str                     # candidate / benchmark / pretrained / fine-tuned
    model: str
    n_samples: int
    runtime_seconds: float
    ms_per_message: float
    parameters_millions: float
    device: str
    metrics: dict = field(default_factory=dict)

    def flat(self) -> dict:
        """Flatten the metrics dict into columns for the CSV / Excel table."""
        row = asdict(self)
        row.update({k: round(v, 4) if isinstance(v, float) else v
                    for k, v in row.pop("metrics").items()})
        row["runtime_seconds"] = round(self.runtime_seconds, 2)
        row["ms_per_message"] = round(self.ms_per_message, 2)
        row["parameters_millions"] = round(self.parameters_millions, 1)
        return row


def save_results(rows: list[ModelResult], path: Path) -> pd.DataFrame:
    table = pd.DataFrame([r.flat() for r in rows])
    table.to_csv(path, index=False, encoding="utf-8")
    print(f"\nSaved {len(rows)} result row(s) -> {path}")
    return table


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def count_parameters_millions(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


class Timer:
    """Context manager: `with Timer() as t: ...` then read t.seconds."""
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._start
        return False


def print_table(table: pd.DataFrame, columns: list[str]) -> None:
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(table[columns].to_string(index=False))
