"""
app.py - Streamlit app: SMS scam triage for a telecom fraud-operations team.

Flow for every message
  1. mask links / phone numbers / e-mails / long numbers (same format as the training data)
  2. Pipeline 1 - fine-tuned text-classification model: scam or safe + confidence
  3. Pipeline 2 - zero-shot-classification model, only for messages flagged as scam:
     which queue (banking, delivery, government, telecom, personal) + confidence
  4. priority + a full-sentence recommended action for the analyst

Run locally:  streamlit run app/app.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Reuse the experiment code (config + masking) instead of copying it, so the
# app and the programs can never disagree on labels or text cleaning.
PROGRAMS_DIR = Path(__file__).resolve().parent.parent / "programs"
sys.path.insert(0, str(PROGRAMS_DIR))
import config  # noqa: E402
import utils   # noqa: E402

# ---------------------------------------------------------------------------
# App constants
# ---------------------------------------------------------------------------
DEFAULT_REVIEW_THRESHOLD = 0.80    # below this confidence a person must check the message
ACCURACY_TARGET = 0.95             # success criterion 1 (held-out accuracy)
MAX_MESSAGE_CHARS = 1000           # longer input is cut (SMS are far shorter)
MAX_BATCH_ROWS = 200               # keeps a batch within the free cloud CPU's limits
SAMPLE_CSV = Path(__file__).resolve().parent / "sample_messages.csv"
RESULTS_FILE = config.RESULTS_DIR / "p1_pretrained_vs_finetuned.csv"

PRIORITY_HIGH = "🔴 High"
PRIORITY_REVIEW = "🟡 Review"
PRIORITY_LOW = "🟢 Low"
PRIORITY_ORDER = {PRIORITY_HIGH: 0, PRIORITY_REVIEW: 1, PRIORITY_LOW: 2}

# One recommended action per scam queue (dict: O(1) look-up by queue name)
QUEUE_ACTIONS = {
    "banking": "Warn the customer not to open the link or share card details or one-time "
               "codes, and to call the number printed on their card. Add the sender to the "
               "bank-impersonation block list.",
    "delivery": "Tell the customer that USPS, UPS and FedEx do not charge redelivery fees by "
                "text. Block the sender and submit the link for takedown.",
    "government": "Remind the customer that the IRS and toll agencies do not demand payment "
                  "by text. Log it as government impersonation for the FTC report.",
    "telecom": "The sender pretends to be a phone carrier, possibly AT&T. Tell the customer "
               "AT&T never asks for an account PIN or passcode by text, and escalate it "
               "for brand-abuse takedown.",
    "personal": "Likely a 'wrong number' or family-impersonation opener. Advise the customer "
                "not to reply or send money, and to call the person on a number they already know.",
}
SAFE_ACTION = "No scam signs found. No action is needed unless the customer reports more concerns."
REVIEW_ACTION = "The model is not confident enough. An analyst should read this message before acting."


# ---------------------------------------------------------------------------
# Model loading (cached: each model is loaded once per server, not per click)
# ---------------------------------------------------------------------------
def classifier_source() -> str:
    """Where the fine-tuned model comes from.

    Order: SCAM_MODEL environment variable (local testing) -> local output of
    finetune.py -> the published model on the Hugging Face Hub.
    """
    override = os.environ.get("SCAM_MODEL")
    if override:
        return override
    local = utils.finetuned_dir(smoke=False)
    if local.exists():
        return str(local)
    return config.HF_MODEL_REPO


@st.cache_resource(show_spinner="Loading the scam classifier…")
def load_classifier(source: str):
    from transformers import pipeline
    return pipeline("text-classification", model=source, device="cpu")


@st.cache_resource(show_spinner="Loading the scam-type tagger…")
def load_tagger(model_id: str):
    from transformers import pipeline
    return pipeline("zero-shot-classification", model=model_id, device="cpu")


@st.cache_data
def held_out_scores() -> dict | None:
    """Fine-tuned model's held-out scores, read from the experiment results."""
    try:
        table = pd.read_csv(RESULTS_FILE, encoding="utf-8")
        row = table[table["role"] == "fine-tuned"].iloc[0]
        return row.to_dict()
    except (FileNotFoundError, IndexError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Triage logic
# ---------------------------------------------------------------------------
DESCRIPTION_TO_QUEUE = {d: q for q, d in config.P2_LABEL_DESCRIPTIONS.items()}


def clean_inputs(texts: list[str]) -> list[str]:
    """Validate and normalise raw messages before they reach the models."""
    cleaned = []
    for text in texts:
        text = "" if pd.isna(text) else str(text).strip()
        cleaned.append(text[:MAX_MESSAGE_CHARS])
    return cleaned


def triage(texts: list[str], classifier, tagger, threshold: float) -> pd.DataFrame:
    masked = [utils.mask_text(t) for t in texts]
    verdicts = classifier(masked, batch_size=config.INFER_BATCH_SIZE,
                          truncation=True, max_length=config.MAX_TOKENS)

    rows = []
    for original, verdict in zip(texts, verdicts):
        rows.append({"Message": original, "Verdict": verdict["label"],
                     "Confidence": verdict["score"], "Scam type": "—",
                     "Type confidence": None})
    table = pd.DataFrame(rows)

    # Pipeline 2 runs only on messages flagged as scam - saves time on safe ones.
    scam_idx = table.index[table["Verdict"] == config.LABEL_SCAM].tolist()
    if scam_idx:
        tags = tagger([masked[i] for i in scam_idx],
                      candidate_labels=list(DESCRIPTION_TO_QUEUE),
                      hypothesis_template=config.P2_HYPOTHESIS)
        tags = tags if isinstance(tags, list) else [tags]
        for i, tag in zip(scam_idx, tags):
            table.at[i, "Scam type"] = DESCRIPTION_TO_QUEUE[tag["labels"][0]]
            table.at[i, "Type confidence"] = tag["scores"][0]

    table["Priority"] = [priority(v, c, threshold) for v, c in zip(table["Verdict"], table["Confidence"])]
    table["Recommended action"] = [action(r) for r in table.to_dict("records")]
    return table


def priority(verdict: str, confidence: float, threshold: float) -> str:
    if confidence < threshold:
        return PRIORITY_REVIEW
    return PRIORITY_HIGH if verdict == config.LABEL_SCAM else PRIORITY_LOW


def action(row: dict) -> str:
    if row["Priority"] == PRIORITY_REVIEW:
        return REVIEW_ACTION
    if row["Verdict"] == config.LABEL_SAFE:
        return SAFE_ACTION
    return QUEUE_ACTIONS[row["Scam type"]]


def summary_sentence(row: dict) -> str:
    """One complete sentence an analyst can act on."""
    pct = f"{row['Confidence']:.0%}"
    if row["Priority"] == PRIORITY_REVIEW:
        return (f"The model leans towards **{row['Verdict']}** but is only {pct} confident, "
                f"so this message needs an analyst's review.")
    if row["Verdict"] == config.LABEL_SAFE:
        return f"This message looks **safe** ({pct} confidence)."
    source = config.P2_LABEL_DESCRIPTIONS[row["Scam type"]]
    return (f"This message is likely a **scam** ({pct} confidence). It appears to pretend to "
            f"come from {source} ({row['Type confidence']:.0%} confidence).")


def display_table(table: pd.DataFrame) -> None:
    shown = table.copy()
    shown["Confidence"] = shown["Confidence"].map("{:.0%}".format)
    shown["Type confidence"] = shown["Type confidence"].map(
        lambda v: "—" if pd.isna(v) else f"{v:.0%}")
    st.dataframe(shown, hide_index=True, use_container_width=True,
                 column_config={"Message": st.column_config.TextColumn(width="large"),
                                "Recommended action": st.column_config.TextColumn(width="large")})


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------
def sidebar() -> float:
    st.sidebar.header("Settings")
    threshold = st.sidebar.slider(
        "Review threshold", 0.50, 0.99, DEFAULT_REVIEW_THRESHOLD, 0.01,
        help="Verdicts below this confidence are sent to an analyst for review.")

    st.sidebar.header("Model performance")
    scores = held_out_scores()
    if scores:
        acc = scores["accuracy"]
        status = "meets" if acc >= ACCURACY_TARGET else "is below"
        st.sidebar.metric("Held-out accuracy", f"{acc:.1%}")
        st.sidebar.write(
            f"On {int(scores['n_samples']):,} real test messages the classifier {status} the "
            f"{ACCURACY_TARGET:.0%} target. It catches {scores['recall_scam']:.1%} of scams and "
            f"wrongly flags {scores['false_positive_rate']:.1%} of safe messages.")
    else:
        st.sidebar.info("Held-out scores appear here after predict_finetuned.py has been run.")

    st.sidebar.header("Models")
    st.sidebar.caption(f"Scam classifier: `{classifier_source()}`")
    st.sidebar.caption(f"Scam-type tagger: `{config.P2_CHOSEN_MODEL}`")
    return threshold


def single_message_tab(classifier, tagger, threshold: float) -> None:
    text = st.text_area("Paste a reported text message", height=120,
                        placeholder="e.g. USPS: Your parcel is on hold. Pay the $1.99 fee at …")
    if st.button("Analyze message", type="primary"):
        cleaned = clean_inputs([text])
        if not cleaned[0]:
            st.warning("Please paste a message first.")
            return
        if len(text.strip()) > MAX_MESSAGE_CHARS:
            st.info(f"Only the first {MAX_MESSAGE_CHARS} characters were analyzed.")
        with st.spinner("Analyzing…"):
            st.session_state["single_result"] = (text, threshold,
                                                 triage(cleaned, classifier, tagger, threshold))

    # Show the last result until the message changes (it survives other clicks).
    stored = st.session_state.get("single_result")
    if stored and stored[0] == text:
        stored_text, stored_threshold, table = stored
        if stored_threshold != threshold:
            st.caption("The review threshold changed - click Analyze again to apply it.")
        row = table.iloc[0].to_dict()
        st.subheader(f"Priority: {row['Priority']}")
        st.markdown(summary_sentence(row))
        st.markdown(f"**Recommended action:** {row['Recommended action']}")
        display_table(table)


def batch_tab(classifier, tagger, threshold: float) -> None:
    st.write(f"Upload a CSV with a **text** column (up to {MAX_BATCH_ROWS} rows). "
             "If it also has a **label** column (scam / safe), the app reports its accuracy.")
    if SAMPLE_CSV.exists():
        st.download_button("Download a sample CSV", SAMPLE_CSV.read_bytes(),
                           file_name="sample_messages.csv", mime="text/csv",
                           on_click="ignore")
    upload = st.file_uploader("CSV file", type="csv")
    if upload is None:
        return

    try:
        data = pd.read_csv(upload, encoding="utf-8")
    except (UnicodeDecodeError, pd.errors.ParserError) as err:
        st.error(f"Could not read the file as a UTF-8 CSV: {err}")
        return
    if "text" not in data.columns:
        st.error(f"The CSV needs a 'text' column; found: {', '.join(data.columns)}")
        return
    if len(data) > MAX_BATCH_ROWS:
        st.warning(f"Only the first {MAX_BATCH_ROWS} of {len(data)} rows are analyzed.")
        data = data.head(MAX_BATCH_ROWS)

    texts = clean_inputs(data["text"].tolist())
    keep = [i for i, t in enumerate(texts) if t]
    if not keep:
        st.error("No non-empty messages found in the 'text' column.")
        return

    # Streamlit re-runs the whole script after every interaction. Keep the last
    # result in session_state so the models only run again when the file or
    # the threshold changes, not on every click.
    run_key = (upload.file_id, threshold)
    if st.session_state.get("batch_key") != run_key:
        with st.spinner(f"Analyzing {len(keep)} messages…"):
            st.session_state["batch_table"] = triage([texts[i] for i in keep],
                                                     classifier, tagger, threshold)
        st.session_state["batch_key"] = run_key
    table = st.session_state["batch_table"].copy()

    counts = table["Priority"].value_counts()
    cols = st.columns(3)
    for col, level in zip(cols, PRIORITY_ORDER):
        col.metric(level, int(counts.get(level, 0)))

    if "label" in data.columns:
        truth = data["label"].iloc[keep].astype(str).str.strip().str.lower().tolist()
        metrics = utils.binary_metrics(truth, table["Verdict"].tolist())
        st.success(
            f"Accuracy on this file: {metrics['accuracy']:.1%} "
            f"({metrics['tp'] + metrics['tn']} of {len(truth)} correct). "
            f"{metrics['fn']} scam(s) were missed and {metrics['fp']} safe message(s) were flagged.")
        table.insert(2, "True label", truth)

    table = table.sort_values("Priority", key=lambda s: s.map(PRIORITY_ORDER), kind="stable")
    display_table(table)
    st.download_button("Download results (CSV)", table.to_csv(index=False).encode("utf-8"),
                       file_name="triage_results.csv", mime="text/csv",
                       on_click="ignore")       # downloading must not re-run the page


def about_tab() -> None:
    st.markdown(f"""
**What it does.** Customers forward suspicious texts to their carrier. This tool
screens each reported message so fraud analysts handle the riskiest ones first.

**How it works.**
1. Links, phone numbers, e-mails and long numbers are replaced with tags such as `<URL>`.
2. A fine-tuned RoBERTa classifier decides **scam** or **safe**.
3. For scams, a zero-shot model (`{config.P2_CHOSEN_MODEL}`) picks the impersonated
   sender type, which sets the recommended action.
4. Verdicts below the review threshold go to an analyst instead of being auto-actioned.

**Limits.** The model reads text only (it does not open links), it was trained mostly on
English messages, and it can still miss new scam styles — high-impact actions should be
confirmed by an analyst.
""")


def main() -> None:
    st.set_page_config(page_title="SMS Scam Triage", page_icon="📱", layout="wide")
    st.title("📱 SMS Scam Triage")
    st.caption("ISOM5160 student project for an AT&T fraud-operations use case. "
               "Not affiliated with or endorsed by AT&T.")
    threshold = sidebar()

    source = classifier_source()
    if source.startswith("YOUR-HF-USERNAME"):
        st.error("No fine-tuned model found. Run finetune.py, or set HF_MODEL_REPO in config.py.")
        st.stop()
    try:
        classifier = load_classifier(source)
        tagger = load_tagger(config.P2_CHOSEN_MODEL)
    except Exception as err:
        st.error(f"A model could not be loaded ({err}). Please try again in a minute.")
        st.stop()

    single, batch, about = st.tabs(["Single message", "Batch (CSV)", "About"])
    with single:
        single_message_tab(classifier, tagger, threshold)
    with batch:
        batch_tab(classifier, tagger, threshold)
    with about:
        about_tab()


if __name__ == "__main__":     # Streamlit runs this file as __main__; importing it
    main()                     # (e.g. from analyze_live_test.py) only loads the functions
