"""
config.py - one place for every path, model name, label and hyper-parameter.

Keeping these as named constants (instead of "magic numbers" inside the
scripts) means a change - e.g. a different base model - is made once here and
every program picks it up.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (pathlib keeps them working on both Windows and macOS)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"

# Smoke-test runs (tiny samples, used to check the code end to end) write to
# separate folders so they never overwrite the real experiment results.
SMOKE_SUBDIR = "smoke"
SMOKE_SAMPLE_SIZE = 64

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_ID = "rasheq/scam-shield-sms"
SPLITS = ("train", "validation", "test")
TEXT_COL = "text"
LABEL_COL = "label"
TYPE_COL = "scam_type"
SOURCE_COL = "source"

# Our two classes. The id order matters: 1 = scam is the "positive" class,
# so precision / recall below are measured for catching scams.
LABEL_SAFE = "safe"
LABEL_SCAM = "scam"
LABEL2ID = {LABEL_SAFE: 0, LABEL_SCAM: 1}
ID2LABEL = {idx: name for name, idx in LABEL2ID.items()}

SEED = 42

# ---------------------------------------------------------------------------
# Pipeline 1 - scam vs safe text classification (the pipeline we fine-tune)
# ---------------------------------------------------------------------------
# A dict maps each candidate model to a dict that translates the model's own
# output labels into our two labels. Dict look-ups are O(1) and make the
# translation explicit, so one evaluation loop works for every model.
P1_CANDIDATES = {
    "mrm8488/bert-tiny-finetuned-sms-spam-detection": {
        "LABEL_0": LABEL_SAFE, "LABEL_1": LABEL_SCAM},
    "mariagrandury/roberta-base-finetuned-sms-spam-detection": {
        "LABEL_0": LABEL_SAFE, "LABEL_1": LABEL_SCAM},
    "Ngadou/bert-sms-spam-dectector": {
        "Normal": LABEL_SAFE, "Spam": LABEL_SCAM},
}

# A published model that was trained on this very dataset. It is NOT a
# fine-tuning candidate (it has already seen our training split); we report it
# only as an outside reference point.
P1_BENCHMARK = {
    "rasheq/scam-shield-distilbert": {"safe": LABEL_SAFE, "scam": LABEL_SCAM},
}

# The candidate picked for fine-tuning after running predict_base.py.
# See results/p1_pretrained_comparison.csv and the README for the reasoning.
P1_CHOSEN_BASE = "mariagrandury/roberta-base-finetuned-sms-spam-detection"

FINETUNED_DIRNAME = "finetuned-scam-classifier"
# Hugging Face Hub repo the fine-tuned model is uploaded to (used by the app).
HF_MODEL_REPO = "Angel426/att-sms-scam-classifier"

# Fine-tuning hyper-parameters
MAX_TOKENS = 128            # SMS median is ~100 characters, so 128 tokens is plenty
TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

# Inference batch size used when timing every model (kept equal for fairness)
INFER_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Pipeline 2 - scam-type tagging with zero-shot classification (no fine-tuning)
# ---------------------------------------------------------------------------
P2_CANDIDATES = [
    "facebook/bart-large-mnli",
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    "typeform/distilbert-base-uncased-mnli",
]
P2_CHOSEN_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"   # confirmed by zeroshot_select.py

# Dataset scam types -> the five routing queues an analyst would use.
# "others" and "unknown" are left out of the evaluation: their ground truth
# does not name a specific type, so no tagger could be scored on them.
P2_TYPE_MAP = {
    "banking": "banking",
    "delivery": "delivery",
    "government": "government",
    "telecom": "telecom",
    "wrong number": "personal",
    "hey mum/dad": "personal",
}

# Each queue gets a plain-English description; the zero-shot model checks
# whether "This text message pretends to come from <description>." is true.
P2_LABEL_DESCRIPTIONS = {
    "banking": "a bank, credit card or payment service",
    "delivery": "a parcel delivery or postal service",
    "government": "a government agency, tax office or toll authority",
    "telecom": "a mobile phone carrier or telecom provider",
    "personal": "a friend, family member or stranger",
}
P2_HYPOTHESIS = "This text message pretends to come from {}."
