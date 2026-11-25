# SMS Scam Triage for AT&T Fraud Operations

ISOM5160 group project (Fall 2026). A student project built around an AT&T
use case; it is not affiliated with or endorsed by AT&T.

Customer-reported text messages are (1) classified as **scam** or **safe** by a
fine-tuned text classifier and (2) routed to a scam-type queue (banking,
delivery, government, telecom, personal) by a zero-shot classifier, so fraud
analysts can handle the riskiest reports first.

| | Pipeline 1 — scam vs safe | Pipeline 2 — scam type |
|---|---|---|
| HF pipeline | `text-classification` | `zero-shot-classification` |
| Candidates compared | `mrm8488/bert-tiny-finetuned-sms-spam-detection`, `mariagrandury/roberta-base-finetuned-sms-spam-detection`, `Ngadou/bert-sms-spam-dectector` (+ benchmark `rasheq/scam-shield-distilbert`) | `facebook/bart-large-mnli`, `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`, `typeform/distilbert-base-uncased-mnli` |
| Fine-tuned? | Yes, on `rasheq/scam-shield-sms` | No |

Dataset: [rasheq/scam-shield-sms](https://huggingface.co/datasets/rasheq/scam-shield-sms) (CC BY 4.0, English).

## Repository layout

```
programs/                 experiment programs (.py)
  config.py               every path, model name and hyper-parameter
  utils.py                shared helpers: seeding, masking, metrics, result records
  classify.py             shared evaluation routine for text classifiers
  prepare_data.py         1. download dataset -> data/*.csv
  predict_base.py         2. compare 3 pretrained classifiers (Pipeline 1)
  finetune.py             3. fine-tune the chosen classifier
  predict_finetuned.py    4. pretrained vs fine-tuned on the same test set
  zeroshot_select.py      5. compare 3 zero-shot taggers (Pipeline 2)
  push_to_hub.py          6. upload the fine-tuned model to Hugging Face
  run_all.py              runs steps 1-5 in order
data/                     dataset CSVs (committed so every run uses the same data)
results/                  experiment tables (CSV/JSON) quoted in the report
app/
  app.py                  Streamlit app (single message, batch CSV, about)
  requirements.txt        app-only packages (CPU PyTorch) used by Streamlit Cloud
  sample_messages.csv     12 realistic US messages with labels, for demos
.streamlit/config.toml    app theme and upload limit
```

## The Streamlit app

For each message the app masks links / numbers, runs the fine-tuned classifier
(scam or safe), runs the zero-shot tagger on scams only, and shows a priority
(🔴 High / 🟡 Review / 🟢 Low), a full-sentence summary and a recommended
action in a table. Batch mode takes a CSV with a `text` column; if a `label`
column is present, the app also reports accuracy on that file (used for the
live-app tests).

The classifier is loaded from, in order: the `SCAM_MODEL` environment
variable, the local `models/finetuned-scam-classifier/`, then
`config.HF_MODEL_REPO` on the Hugging Face Hub (what Streamlit Cloud uses).

```bash
streamlit run app/app.py
```

Streamlit Cloud settings: main file `app/app.py`; packages come from `app/requirements.txt`.

## Running the experiments on the Windows GPU machine

Requires Python 3.11–3.13 and an NVIDIA driver. RTX 50-series cards need the
**CUDA 12.8 (cu128)** build of PyTorch; older builds fail with "no kernel image".

```bash
git clone <this repo URL>
cd sms-scam-triage
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-train.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True` and the GPU name. Then:

```bash
cd programs
python run_all.py --smoke     # ~5 min: checks every step on 64 messages
python run_all.py             # full run; results go to results/
```

Upload the model (needs a Hugging Face token with write access; set
`HF_MODEL_REPO` in `config.py` to your username first):

```bash
hf auth login
python push_to_hub.py
```

Commit and push `results/` afterwards so the report uses the numbers from this run.

On macOS use `source .venv/bin/activate` and a plain `pip install torch`.

## Where the reported numbers come from

* **Fine-tuning** ran on an NVIDIA RTX 5070 (`results/finetune_log.json`, `results/full_run_log.txt`).
* **All evaluation and runtime comparisons** (`p1_*`, `p2_*` result files) were re-run on one
  machine, an Apple M1 Pro (MPS), so runtimes in every table are comparable
  (`results/mac_eval_log.txt`). The cloud app runs on CPU, so these laptop
  figures are closer to deployment speed than GPU figures.
* Accuracy figures match the GPU run exactly for Pipeline 1; for Pipeline 2 one of 737
  messages flipped (0.8168 → 0.8182), a floating-point difference between GPUs.
* `predict_finetuned.py` uses the local `models/finetuned-scam-classifier/` if present,
  otherwise downloads the identical published model `Angel426/att-sms-scam-classifier`,
  so it can be re-run without re-training.

## Reproducibility

* All randomness is seeded (`config.SEED = 42`); test-set sampling uses the same seed.
* Every runtime figure is saved next to the hardware it was measured on.
* `--smoke` runs write to `results/smoke/` and `models/smoke/`, never to the real outputs.

## Data-structure choices

* **pandas DataFrame** for each split — a table of messages and labels with fast filtering, sampling and CSV export.
* **dict** for label translations (each model's own label → our `safe`/`scam`) and metric bundles — O(1) look-up by name, so one evaluation loop serves every model.
* **dataclass `ModelResult`** — every experiment row has the same fixed fields, so rows from different models line up in one table.
* **list** of candidate models — the order of the systematic comparison loop.

## Secrets

The Hugging Face token is only needed to upload the model. It is entered with
`hf auth login` and never stored in the repository. The public app downloads
the public model and needs no secret.
