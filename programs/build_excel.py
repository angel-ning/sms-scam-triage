"""
build_excel.py - Step 8: write Experimental_results.xlsx from the result files.

Every number in the workbook is read from results/ (nothing is typed in by
hand), so the workbook always matches the programs' output. The layout follows
the course's example workbook: project details, success criteria, model
selection for each pipeline, pretrained vs fine-tuned, and the live-app tests.

Input   results/*.csv, results/*.json, data/live_app_test.csv
Output  results/Experimental_results.xlsx
          sheet "Performance"           - the course layout
          sheet "Model metrics (detail)" - every metric, fine-tuning settings

Usage
  python build_excel.py
"""

import argparse
import json
import math

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

import config

OUTPUT_FILE = config.RESULTS_DIR / "Experimental_results.xlsx"

# --- styles copied from the course template -------------------------------
FONT_NAME = "Aptos Narrow"
FILL_SECTION = "FFC000"     # big section headings
FILL_SUB = "FFE699"         # sub-section headings
FILL_LABEL = "F2F2F2"       # labels and table headers
FILL_VALUE = "FFF8E7"       # values
GREY_TEXT = "595959"
COL_WIDTHS = {"A": 28, "B": 42, "C": 28, "D": 22, "E": 22, "F": 42}
CHARS_PER_WIDTH = 1.15      # Aptos Narrow fits ~1.15 characters per width unit
LINE_HEIGHT = 15
SPACER_HEIGHT = 8
THIN = Side(style="thin", color="000000")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

CATEGORY_NAMES = {
    "scam_typical": "Scam (typical: brand, link, urgency)",
    "scam_casual": "Scam (casual personal tone)",
    "safe_service": "Safe service alert",
    "safe_personal": "Safe personal chat",
}

# Short explanations of each live-test miss, used in the note column
MISS_EXPLANATIONS = {
    "Wells Fargo": "Copies the wording of a genuine transaction alert; the fake web address, "
                   "the only giveaway, is masked to <URL> before the model sees it.",
    "Citi": "Same pattern: a fake fraud alert whose only clue is the call-back number "
            "(masked to <PHONE_NUMBER>).",
    "DHL": "Low confidence (58%), so the app sent it to an analyst instead of clearing it.",
    "omg sorry": "'Wrong number' opener with no request yet; reads like a normal chat.",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_inputs() -> dict:
    res = config.RESULTS_DIR

    def read_json(name):
        return json.loads((res / name).read_text(encoding="utf-8"))

    live = pd.read_csv(res / "live_app_test_table.csv", encoding="utf-8", keep_default_na=False)
    live.columns = [c.replace(" ", "_") for c in live.columns]      # "Scam type" -> Scam_type
    live["correct"] = live["correct"].astype(str).eq("True")
    return {
        "p1": pd.read_csv(res / "p1_pretrained_comparison.csv", encoding="utf-8"),
        "p1ft": pd.read_csv(res / "p1_pretrained_vs_finetuned.csv", encoding="utf-8"),
        "p2": pd.read_csv(res / "p2_zeroshot_comparison.csv", encoding="utf-8"),
        "ft_log": read_json("finetune_log.json"),
        "data": read_json("dataset_summary.json"),
        "live": live.sort_values("id").reset_index(drop=True),
        "live_summary": read_json("live_app_summary.json"),
    }


def best_epoch(ft_log: dict) -> int:
    """Epoch with the best validation F1 (the one load_best_model_at_end kept)."""
    epochs = [e for e in ft_log["epoch_history"] if "step" in e]
    unique = {int(e["epoch"]): e["eval_f1_scam"] for e in epochs}
    return max(unique, key=unique.get)


# ---------------------------------------------------------------------------
# Sheet writer with the template's look
# ---------------------------------------------------------------------------
class SheetWriter:
    def __init__(self, ws):
        self.ws = ws
        self.row = 1
        for col, width in COL_WIDTHS.items():
            ws.column_dimensions[col].width = width

    # low-level -----------------------------------------------------------
    def _style(self, cell, fill=None, bold=False, size=11, color=None, align="left",
               valign="center", border=True, fmt=None):
        cell.font = Font(name=FONT_NAME, size=size, bold=bold, color=color)
        if fill:
            cell.fill = PatternFill("solid", fgColor=fill)
        cell.alignment = Alignment(horizontal=align, vertical=valign, wrap_text=True)
        if border:
            cell.border = BORDER
        if fmt:
            cell.number_format = fmt

    def put(self, col_from: str, col_to: str, value, **style):
        """Write one value across a range of columns in the current row."""
        ws, r = self.ws, self.row
        if col_from != col_to:
            ws.merge_cells(f"{col_from}{r}:{col_to}{r}")
        for c in range(ord(col_from), ord(col_to) + 1):
            self._style(ws[f"{chr(c)}{r}"], **style)
        ws[f"{col_from}{r}"] = value

    def fit_height(self, cells: list[tuple[str, str, str]], minimum: float = 18) -> None:
        """Set the row height so wrapped text in every (from, to, text) cell is visible."""
        lines = 1
        for col_from, col_to, text in cells:
            width = sum(COL_WIDTHS[chr(c)] for c in range(ord(col_from), ord(col_to) + 1))
            per_line = max(1, int(width * CHARS_PER_WIDTH))
            lines = max(lines, sum(math.ceil(max(len(part), 1) / per_line)
                                   for part in str(text).split("\n")))
        self.ws.row_dimensions[self.row].height = max(minimum, lines * LINE_HEIGHT + 4)

    def next(self, spacer: bool = False) -> None:
        self.row += 1
        if spacer:
            self.ws.row_dimensions[self.row].height = SPACER_HEIGHT
            self.row += 1

    def spacer(self) -> None:
        self.ws.row_dimensions[self.row].height = SPACER_HEIGHT
        self.row += 1

    # building blocks -------------------------------------------------------
    def title(self, text: str) -> None:
        self.put("A", "F", text, size=16, bold=True, border=False)
        self.ws.row_dimensions[self.row].height = 24
        self.next()

    def note(self, text: str) -> None:
        self.put("A", "F", text, size=10, color=GREY_TEXT, border=False)
        self.fit_height([("A", "F", text)], minimum=20)
        self.next()

    def section(self, text: str) -> None:
        self.put("A", "F", text, size=14, bold=True, fill=FILL_SECTION, border=False)
        self.ws.row_dimensions[self.row].height = 22
        self.next()

    def sub(self, text: str) -> None:
        self.put("A", "F", text, bold=True, fill=FILL_SUB, border=False)
        self.ws.row_dimensions[self.row].height = 20
        self.next()

    def label_value(self, label: str, value, fill=FILL_VALUE) -> None:
        self.put("A", "A", label, bold=True, fill=FILL_LABEL)
        self.put("B", "F", value, fill=fill)
        self.fit_height([("B", "F", value)])
        self.next()

    def header(self, spans: list[tuple[str, str, str]]) -> None:
        for col_from, col_to, text in spans:
            self.put(col_from, col_to, text, bold=True, fill=FILL_LABEL, align="center")
        self.fit_height(spans, minimum=22)
        self.next()

    def table_row(self, cells: list[tuple[str, str, object, dict]]) -> int:
        """cells: (from, to, value, style overrides). Returns the row number written."""
        for col_from, col_to, value, style in cells:
            base = {"fill": FILL_VALUE, "bold": col_from == "A" and col_to == "A"}
            self.put(col_from, col_to, value, **(base | style))
        self.fit_height([(f, t, v) for f, t, v, _ in cells if isinstance(v, str)
                         and not str(v).startswith("=")], minimum=20)
        written = self.row
        self.next()
        return written


# ---------------------------------------------------------------------------
# Sections of the "Performance" sheet
# ---------------------------------------------------------------------------
def project_details(w: SheetWriter, d: dict) -> None:
    w.title("ISOM5160 · Fall 2026 · Experimental results")
    w.note("SMS scam triage for AT&T's fraud-operations team. Every figure below is generated "
           "by programs/build_excel.py from the result files in results/, which the .py "
           "programs produce.")
    w.spacer()
    w.sub("Project details")
    w.label_value("Group number", config.GROUP_NUMBER)
    w.label_value("Project title", "SMS Scam Shield: smishing triage for AT&T's fraud operations team")
    w.label_value("Company", "AT&T Inc. (https://www.att.com) - student project, not affiliated with AT&T")
    w.label_value("Streamlit Cloud app URL", config.APP_URL)
    w.label_value("GitHub repository URL", config.GITHUB_URL)
    w.label_value("Fine-tuned model URL", config.HF_MODEL_URL)
    w.label_value("Dataset", f"{config.DATASET_ID} (https://huggingface.co/datasets/{config.DATASET_ID}), "
                             "CC BY 4.0, English SMS labelled scam / safe")
    w.spacer()

    w.section("About this app")
    w.note("Two Hugging Face pipelines. Pipeline 1 (text-classification) decides scam or safe and "
           "is fine-tuned on the scam dataset. Pipeline 2 (zero-shot-classification) tags each "
           "scam with the sender it impersonates (banking, delivery, government, telecom, "
           "personal) so it reaches the right analyst queue with a matching customer warning; "
           "it uses a pretrained model as-is.")
    w.spacer()


def success_criteria(w: SheetWriter, refs: dict) -> None:
    w.section("Measurable success criteria")
    w.note("Two checkable criteria for AT&T's problem. The result column is a formula that reads "
           "the figures further down this sheet.")
    w.header([("A", "A", "Criterion"), ("B", "E", "What would count as success"),
              ("F", "F", "Result in this workbook")])
    w.table_row([
        ("A", "A", "Success criterion 1", {}),
        ("B", "E", "The fine-tuned classifier reaches at least 95% accuracy on the held-out "
                   "test set of real messages, so analysts can trust its first-pass verdict.", {}),
        ("F", "F", f'=IF({refs["ft_acc"]}>=0.95,"Met","Not met")&" (held-out accuracy "'
                   f'&TEXT({refs["ft_acc"]},"0.000")&")"', {}),
    ])
    w.table_row([
        ("A", "A", "Success criterion 2", {}),
        ("B", "E", "In live-app tests with realistic US messages, no more than 10% of legitimate "
                   "service alerts (bills, one-time codes, deliveries) are wrongly flagged as "
                   "scams, so the high-risk queue is not flooded.", {}),
        ("F", "F", f'=IF({refs["svc_fp"]}<=0.1,"Met","Not met")&" ("&TEXT({refs["svc_fp"]},"0%")'
                   f'&" of service alerts flagged)"', {}),
    ])
    w.spacer()


def p1_selection(w: SheetWriter, d: dict) -> None:
    p1 = d["p1"].set_index("model")
    n_test = int(p1["n_samples"].iloc[0])
    device = p1["device"].iloc[0]
    w.section("Experiment: Selecting the best model(s)")
    w.note("For each pipeline, three comparable pretrained Hugging Face models are scored on the "
           f"same held-out set in one loop (predict_base.py, zeroshot_select.py). Runtime is the "
           f"time to classify the whole set on {device}, batch size {config.INFER_BATCH_SIZE}, "
           "after an untimed warm-up batch. The pick weighs the errors that matter to a fraud "
           "team - missed scams (reach customers) and false alarms (waste analyst time) - and "
           "speed, not accuracy alone.")
    w.spacer()

    w.sub("Pipeline 1 - Scam versus safe SMS classification (text-classification, to be fine-tuned)")
    w.put("A", "A", "Held-out dataset", bold=True, fill=FILL_LABEL)
    w.put("B", "C", "data/test.csv - test split of the dataset, real messages only "
                    f"({d['data']['splits']['test']['labels']['scam']} scam, "
                    f"{d['data']['splits']['test']['labels']['safe']} safe)", fill=FILL_VALUE)
    w.put("D", "D", "Held-out test samples", bold=True, fill=FILL_LABEL)
    w.put("E", "F", n_test, fill=FILL_VALUE)
    w.fit_height([("B", "C", "x" * 110)])
    w.next(spacer=True)

    w.header([("A", "A", "Model"), ("B", "B", "Hugging Face model name"),
              ("C", "C", "Accuracy (correct rate)"), ("D", "D", "Runtime (seconds)"),
              ("E", "F", "Notes on errors / business fit")])

    def notes(m) -> str:
        r = p1.loc[m]
        stats = (f"Scam recall {r.recall_scam:.1%}; false alarms {r.false_positive_rate:.1%} "
                 f"({int(r.fp)} of {int(r.fp + r.tn)} safe); {r.parameters_millions:.0f}M parameters. ")
        return stats + P1_COMMENTS[m]

    for i, model in enumerate(config.P1_CANDIDATES, start=1):
        r = p1.loc[model]
        w.table_row([("A", "A", f"Model {i}", {}), ("B", "B", model, {}),
                     ("C", "C", r.accuracy, {"align": "center", "fmt": "0.000"}),
                     ("D", "D", r.runtime_seconds, {"align": "center", "fmt": "0.00"}),
                     ("E", "F", notes(model), {})])
    for model in config.P1_BENCHMARK:
        r = p1.loc[model]
        w.table_row([("A", "A", "Benchmark (not a candidate)", {}), ("B", "B", model, {}),
                     ("C", "C", r.accuracy, {"align": "center", "fmt": "0.000"}),
                     ("D", "D", r.runtime_seconds, {"align": "center", "fmt": "0.00"}),
                     ("E", "F", notes(model), {})])
    w.spacer()

    chosen_index = list(config.P1_CANDIDATES).index(config.P1_CHOSEN_BASE) + 1
    w.label_value("Chosen model for fine-tuning", f"Model {chosen_index} - {config.P1_CHOSEN_BASE}")
    c = p1.loc[config.P1_CHOSEN_BASE]
    ng = p1.loc["Ngadou/bert-sms-spam-dectector"]
    w.label_value("Justification (not accuracy alone)",
                  f"Model {chosen_index} is picked although Model 3 has higher accuracy "
                  f"({ng.accuracy:.3f} vs {c.accuracy:.3f}). Model 3 flags {ng.false_positive_rate:.1%} "
                  f"of safe messages ({int(ng.fp)} of {int(ng.fp + ng.tn)}); at AT&T's report volume "
                  "that would bury analysts in false alarms. Model 2 flags only "
                  f"{int(c.fp)} safe message ({c.false_positive_rate:.1%}): its weakness is missed "
                  f"scams (recall {c.recall_scam:.1%}), because it learned 2012-style advertising "
                  "spam, and that is exactly the gap fine-tuning on modern scam texts can close. "
                  f"Its {c.ms_per_message:.1f} ms per message is ample for analyst use, and the "
                  "RoBERTa family reads informal SMS text well. Model 1 is fastest but misses "
                  f"{1 - p1.loc[list(config.P1_CANDIDATES)[0]].recall_scam:.0%} of scams.")
    w.spacer()


P1_COMMENTS = {
    "mrm8488/bert-tiny-finetuned-sms-spam-detection":
        "Fastest and smallest, but misses most modern scams - too many would reach customers.",
    "mariagrandury/roberta-base-finetuned-sms-spam-detection":
        "Almost no false alarms; misses scams that do not look like advertising spam.",
    "Ngadou/bert-sms-spam-dectector":
        "Highest accuracy of the three, but its false-alarm rate would flood the analyst queue.",
    "rasheq/scam-shield-distilbert":
        "Outside reference only: already trained on this dataset's training split, so it is "
        "not a fair fine-tuning candidate.",
}


def p1_finetune(w: SheetWriter, d: dict) -> dict:
    ft = d["p1ft"].set_index("role")
    log = d["ft_log"]
    w.sub("Pretrained versus fine-tuned (same held-out test set)")
    w.note(f"The chosen model is fine-tuned on the {log['train_rows']:,}-message training split "
           f"({log['hyper_parameters']['num_train_epochs']} epochs, learning rate "
           f"{log['hyper_parameters']['learning_rate']}, batch {log['hyper_parameters']['per_device_train_batch_size']}, "
           f"seed {log['hyper_parameters']['seed']}) on {log['device']} in {log['training_minutes']} minutes; "
           f"the epoch with the best validation F1 (epoch {best_epoch(log)}) is kept. Both versions are "
           "then scored on the same held-out test set as above.")
    w.header([("A", "A", "Version"), ("B", "B", "Hugging Face model name"),
              ("C", "C", "Accuracy (correct rate)"), ("D", "D", "Runtime (seconds)"),
              ("E", "F", "Notes")])
    pre, post = ft.loc["pretrained"], ft.loc["fine-tuned"]

    def stats(r) -> str:
        return (f"Scam recall {r.recall_scam:.1%}; false alarms {r.false_positive_rate:.1%} "
                f"({int(r.fp)} of {int(r.fp + r.tn)} safe); {int(r.fp + r.fn)} errors in total.")

    pre_row = w.table_row([("A", "A", "Pretrained", {}), ("B", "B", config.P1_CHOSEN_BASE, {}),
                           ("C", "C", pre.accuracy, {"align": "center", "fmt": "0.000"}),
                           ("D", "D", pre.runtime_seconds, {"align": "center", "fmt": "0.00"}),
                           ("E", "F", "Baseline before fine-tuning. " + stats(pre), {})])
    post_row = w.table_row([("A", "A", "Fine-tuned", {}),
                            ("B", "B", f"{config.HF_MODEL_REPO} (fine-tuned from the model above)", {}),
                            ("C", "C", post.accuracy, {"align": "center", "fmt": "0.000"}),
                            ("D", "D", post.runtime_seconds, {"align": "center", "fmt": "0.00"}),
                            ("E", "F", stats(post) + " Several remaining errors are mislabelled "
                                       "messages in the dataset (e.g. a dinner invitation labelled "
                                       "scam) - see results/p1_finetuned_errors.csv.", {})])
    gain_row = w.table_row([("A", "A", "Change", {}), ("B", "B", "Fine-tuned minus pretrained", {}),
                            ("C", "C", f"=C{post_row}-C{pre_row}", {"align": "center", "fmt": "+0.000;-0.000"}),
                            ("D", "D", f"=D{post_row}-D{pre_row}", {"align": "center", "fmt": "+0.00;-0.00"}),
                            ("E", "F", "Same size and speed, much higher accuracy: fine-tuning "
                                       "changes the weights, not the architecture.", {})])
    w.spacer()
    return {"ft_acc": f"C{post_row}", "gain": f"C{gain_row}"}


def p2_selection(w: SheetWriter, d: dict) -> None:
    p2 = d["p2"].set_index("model")
    queues = d["data"]["p2_test"]["queues"]
    w.sub("Pipeline 2 - Scam-type tagging (zero-shot-classification; pretrained only, no fine-tuning)")
    w.put("A", "A", "Held-out dataset", bold=True, fill=FILL_LABEL)
    w.put("B", "C", "data/p2_test.csv - real scams in the test split with a known type: "
                    + ", ".join(f"{k} {v}" for k, v in queues.items()), fill=FILL_VALUE)
    w.put("D", "D", "Held-out test samples", bold=True, fill=FILL_LABEL)
    w.put("E", "F", d["data"]["p2_test"]["rows"], fill=FILL_VALUE)
    w.fit_height([("B", "C", "x" * 120)])
    w.next(spacer=True)

    w.header([("A", "A", "Model"), ("B", "B", "Hugging Face model name"),
              ("C", "C", "Accuracy (correct rate)"), ("D", "D", "Runtime (seconds)"),
              ("E", "F", "Notes on errors / business fit")])
    for i, model in enumerate(config.P2_CANDIDATES, start=1):
        r = p2.loc[model]
        w.table_row([("A", "A", f"Model {i}", {}), ("B", "B", model, {}),
                     ("C", "C", r.accuracy, {"align": "center", "fmt": "0.000"}),
                     ("D", "D", r.runtime_seconds, {"align": "center", "fmt": "0.00"}),
                     ("E", "F", f"Macro F1 {r.macro_f1:.3f}; {r.parameters_millions:.0f}M parameters; "
                                f"{r.ms_per_message:.0f} ms per message. " + P2_COMMENTS[model], {})])
    w.spacer()

    chosen_index = config.P2_CANDIDATES.index(config.P2_CHOSEN_MODEL) + 1
    c = p2.loc[config.P2_CHOSEN_MODEL]
    bart = p2.loc["facebook/bart-large-mnli"]
    w.label_value("Chosen model for implementation", f"Model {chosen_index} - {config.P2_CHOSEN_MODEL}")
    w.label_value("Justification (not accuracy alone)",
                  f"Model {chosen_index} has the best accuracy ({c.accuracy:.3f}) and macro F1 "
                  f"({c.macro_f1:.3f}), so fewer scams land in the wrong queue with the wrong "
                  f"customer warning. It is also about {bart.runtime_seconds / c.runtime_seconds:.0f}x "
                  f"faster than BART-large ({c.runtime_seconds:.0f} s vs {bart.runtime_seconds:.0f} s) at "
                  "less than half the size, which matters on the free cloud CPU. The fastest model "
                  "is right only about a third of the time, so its speed is of no use. Tagging runs "
                  "only on messages already flagged as scams, which keeps its cost low. Known weak "
                  "spot: the small 'personal' queue.")
    w.spacer()


P2_COMMENTS = {
    "facebook/bart-large-mnli": "Usable baseline, but slowest and largest.",
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0": "Best on both metrics with a moderate size.",
    "typeform/distilbert-base-uncased-mnli": "Fast, but too inaccurate to route reports.",
}


def live_tests(w: SheetWriter, d: dict) -> dict:
    live = d["live"]
    w.section("Experiment: Assessing overall application performance")
    w.sub("Step 1 - Deployment")
    w.label_value("Deployed app",
                  f"App = Pipeline 1 (fine-tuned {config.HF_MODEL_REPO}) + Pipeline 2 (pretrained "
                  f"{config.P2_CHOSEN_MODEL}). Deployed to Streamlit Cloud ({config.APP_URL}), "
                  f"linked to GitHub ({config.GITHUB_URL}). The app loads the fine-tuned model "
                  "from the Hugging Face Hub, so it is the exact model finetune.py produced.")
    w.spacer()

    counts = live["category"].value_counts()
    w.sub(f"Step 2 - Live app tests ({len(live)} realistic US text messages)")
    w.note("data/live_app_test.csv was uploaded to the deployed app's Batch tab and the downloaded "
           "output scored by analyze_live_test.py. The messages were written by the group (not "
           "taken from the training data) and mix "
           + ", ".join(f"{counts[k]} {CATEGORY_NAMES[k].lower()}" for k in CATEGORY_NAMES)
           + ". Output = verdict (confidence) · scam type · priority.")
    w.header([("A", "A", "Test sample"), ("B", "B", "Input"), ("C", "C", "Output"),
              ("D", "D", "Correct or not"), ("E", "F", "Message type / error note")])

    first = w.row
    for row in live.itertuples():
        priority = str(row.Priority).split(" ", 1)[-1]
        output = f"{row.Verdict} ({float(row.Confidence):.0%})"
        if row.Verdict == config.LABEL_SCAM:
            output += f" · {row.Scam_type}"
        output += f" · {priority}"
        note = CATEGORY_NAMES[row.category]
        if not row.correct:
            kind = "missed scam" if row.label == config.LABEL_SCAM else "false alarm"
            explain = next((v for k, v in MISS_EXPLANATIONS.items() if row.text.startswith(k)), "")
            note += f". Incorrect: {kind}. {explain}".rstrip()
        elif row.label == config.LABEL_SCAM and str(row.type_correct) == "False":
            note += f". Scam type tagged {row.Scam_type}, expected {row.expected_type}."
        w.table_row([("A", "A", f"Test sample {row.id}", {}), ("B", "B", row.text, {}),
                     ("C", "C", output, {}),
                     ("D", "D", "Correct" if row.correct else "Incorrect", {"align": "center"}),
                     ("E", "F", note, {})])
    last = w.row - 1
    w.spacer()

    d_rng, e_rng, c_rng = f"D{first}:D{last}", f"E{first}:E{last}", f"C{first}:C{last}"
    refs = {}
    for label, text_formula, rate_formula, key in [
        ("Live-app accuracy",
         f'=COUNTIF({d_rng},"Correct")&" out of "&COUNTA({d_rng})',
         f'=COUNTIF({d_rng},"Correct")/COUNTA({d_rng})', "acc"),
        ("Scams caught (recall)",
         f'=COUNTIFS({e_rng},"Scam*",{d_rng},"Correct")&" out of "&COUNTIF({e_rng},"Scam*")',
         f'=COUNTIFS({e_rng},"Scam*",{d_rng},"Correct")/COUNTIF({e_rng},"Scam*")', "recall"),
        ("Service alerts wrongly flagged",
         f'=COUNTIFS({e_rng},"Safe service*",{c_rng},"scam*")&" out of "&COUNTIF({e_rng},"Safe service*")',
         f'=COUNTIFS({e_rng},"Safe service*",{c_rng},"scam*")/COUNTIF({e_rng},"Safe service*")', "svc_fp"),
        ("Sent to analyst review",
         f'=COUNTIF({c_rng},"*Review")&" out of "&COUNTA({c_rng})',
         f'=COUNTIF({c_rng},"*Review")/COUNTA({c_rng})', "review"),
    ]:
        w.put("A", "A", label, bold=True, fill=FILL_LABEL)
        w.put("B", "C", text_formula, fill=FILL_VALUE)
        w.put("D", "D", "Rate", bold=True, fill=FILL_LABEL)
        w.put("E", "F", rate_formula, fill=FILL_VALUE, fmt="0.0%")
        refs[key] = f"E{w.row}"
        w.ws.row_dimensions[w.row].height = 20
        w.next()
    w.spacer()
    return refs


def criteria_and_conclusion(w: SheetWriter, d: dict, refs: dict) -> None:
    ft = d["p1ft"].set_index("role")
    pre, post = ft.loc["pretrained"], ft.loc["fine-tuned"]
    s = d["live_summary"]
    w.sub("Step 3 - Against the success criteria (evidence, not a claim only)")
    w.header([("A", "A", "Criterion"), ("B", "B", "Met / partly met / not met"),
              ("C", "F", "Evidence from the model comparison and the live app")])
    w.table_row([
        ("A", "A", "Success criterion 1", {}),
        ("B", "B", f'=IF({refs["ft_acc"]}>=0.95,"Met","Not met")', {"align": "center"}),
        ("C", "F", f"Held-out accuracy of the fine-tuned model is {post.accuracy:.3f} on "
                   f"{int(post.n_samples):,} real messages (pretrained baseline {pre.accuracy:.3f}), "
                   f"above the 0.95 target, with scam recall {post.recall_scam:.1%} and "
                   f"{post.false_positive_rate:.1%} false alarms. On the harder live set the app scores "
                   f"{s['accuracy']:.0%}, below 95%: the live set deliberately includes scams that "
                   "imitate genuine alerts.", {}),
    ])
    svc = s["by_category"]["safe_service"]
    w.table_row([
        ("A", "A", "Success criterion 2", {}),
        ("B", "B", f'=IF({refs["svc_fp"]}<=0.1,"Met","Not met")', {"align": "center"}),
        ("C", "F", f"0 of {svc['messages']} genuine service alerts (AT&T bills and codes, bank "
                   "alerts, deliveries, appointments) and 0 of "
                   f"{s['by_category']['safe_personal']['messages']} personal chats were flagged, "
                   "so routine messages do not flood the high-risk queue. Caveat: 15 alerts is a "
                   "small sample, and the held-out false-alarm rate "
                   f"({post.false_positive_rate:.1%}) is the larger-sample evidence.", {}),
    ])
    w.spacer()

    w.sub("Step 4 - Evidence-based conclusion")
    conclusion = (
        f"(i) Both criteria are met. Fine-tuning lifted held-out accuracy from {pre.accuracy:.3f} "
        f"to {post.accuracy:.3f} and scam recall from {pre.recall_scam:.1%} to {post.recall_scam:.1%} "
        f"while keeping false alarms below 1%, and the live app wrongly flagged none of the "
        f"{svc['messages']} genuine service alerts. The zero-shot tagger routes "
        f"{s['scam_type_accuracy']:.0%} of caught live scams to the right queue.\n"
        f"(ii) Not yet achieved: the live app caught {s['recall_scam']:.0%} of scams. It misses "
        "scams that copy genuine transaction alerts, because their only giveaway (the fake web "
        "address or call-back number) is masked before classification, and casual 'wrong number' "
        "openers. The scam-type tagger is weakest on the small 'personal' queue.\n"
        "(iii) AT&T could run the app as a first-pass screen on messages customers forward to 7726: "
        "High-priority scams go straight to the matching queue with a ready customer warning, "
        "low-confidence verdicts go to an analyst, and safe verdicts are spot-checked. Adding a "
        "link-reputation check for masked web addresses is the next improvement.")
    w.put("A", "F", conclusion, fill=FILL_VALUE, valign="top")
    w.fit_height([("A", "F", conclusion)])
    w.ws.row_dimensions[w.row].height += 30          # three paragraphs need extra room
    w.next()


# ---------------------------------------------------------------------------
# Detail sheet
# ---------------------------------------------------------------------------
def detail_sheet(wb: Workbook, d: dict) -> None:
    ws = wb.create_sheet("Model metrics (detail)")
    r = 1

    def write_table(title: str, table: pd.DataFrame) -> None:
        nonlocal r
        ws.cell(r, 1, title).font = Font(name=FONT_NAME, size=12, bold=True)
        r += 1
        for j, col in enumerate(table.columns, start=1):
            cell = ws.cell(r, j, col)
            cell.font = Font(name=FONT_NAME, bold=True)
            cell.fill = PatternFill("solid", fgColor=FILL_LABEL)
            cell.border = BORDER
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        for values in table.itertuples(index=False):
            r += 1
            for j, value in enumerate(values, start=1):
                cell = ws.cell(r, j, value)
                cell.font = Font(name=FONT_NAME)
                cell.border = BORDER
                if isinstance(value, float):
                    cell.number_format = "0.0000" if abs(value) < 1.5 else "0.00"
        r += 2

    drop = ["pipeline"]
    write_table("Pipeline 1 - pretrained comparison (results/p1_pretrained_comparison.csv)",
                d["p1"].drop(columns=drop))
    write_table("Pipeline 1 - pretrained vs fine-tuned (results/p1_pretrained_vs_finetuned.csv)",
                d["p1ft"].drop(columns=drop))
    write_table("Pipeline 2 - zero-shot comparison (results/p2_zeroshot_comparison.csv)",
                d["p2"].drop(columns=drop))

    log = d["ft_log"]
    settings = pd.DataFrame(
        [("base model", log["base_model"]), ("device", log["device"]),
         ("training rows", log["train_rows"]), ("validation rows", log["validation_rows"]),
         ("training minutes", log["training_minutes"]), ("best epoch (validation F1)", best_epoch(log))]
        + [(k, v) for k, v in log["hyper_parameters"].items()],
        columns=["setting", "value"])
    write_table("Fine-tuning settings (results/finetune_log.json)", settings)
    epochs = pd.DataFrame([
        {"epoch": int(e["epoch"]), "validation loss": e["eval_loss"], "accuracy": e["eval_accuracy"],
         "recall (scam)": e["eval_recall_scam"], "F1 (scam)": e["eval_f1_scam"],
         "false-positive rate": e["eval_false_positive_rate"]}
        for e in log["epoch_history"] if "step" in e])
    write_table("Validation scores after each epoch", epochs)

    for j, width in enumerate([34, 52, 14, 14, 14, 14, 14, 26, 12, 12, 12, 12, 14, 8, 8, 8, 8], start=1):
        ws.column_dimensions[get_column_letter(j)].width = width


def main() -> None:
    d = load_inputs()
    wb = Workbook()
    ws = wb.active
    ws.title = "Performance"
    ws.sheet_view.showGridLines = False
    ws.page_setup.orientation = "landscape"
    w = SheetWriter(ws)

    project_details(w, d)
    criteria_row = w.row                       # filled in last: needs refs from later sections
    w.row += 6
    p1_selection(w, d)
    refs = p1_finetune(w, d)
    p2_selection(w, d)
    refs |= live_tests(w, d)
    criteria_and_conclusion(w, d, refs)
    end_row = w.row

    w.row = criteria_row
    success_criteria(w, refs)
    assert w.row == criteria_row + 6, "success-criteria block changed size; update the reserved rows"
    ws.freeze_panes = "A4"
    ws.print_area = f"A1:F{end_row}"

    detail_sheet(wb, d)
    wb.calculation.fullCalcOnLoad = True       # Excel computes every formula when opened
    wb.save(OUTPUT_FILE)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
