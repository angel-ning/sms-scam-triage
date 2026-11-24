"""
finetune.py - Step 3: fine-tune the chosen Pipeline 1 model on the scam dataset.

pretrained model (config.P1_CHOSEN_BASE) + training split  ->  fine-tuned model

The model is checked on the validation split after every epoch, and the
epoch with the best scam F1 is kept (the test split is never used here, so
the later test-set comparison stays a fair held-out evaluation).

Output
  models/finetuned-scam-classifier/          (model + tokenizer, used by the app)
  results/finetune_log.json                  (hyper-parameters, per-epoch scores, time)

Usage
  python finetune.py            # full training (run on the GPU machine)
  python finetune.py --smoke    # 64 messages, 1 epoch, quick code check
"""

import argparse

import numpy as np
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

import config
import utils


def to_dataset(df, tokenizer) -> Dataset:
    """DataFrame -> tokenised Hugging Face Dataset with integer labels."""
    data = Dataset.from_dict({
        "text": df[config.TEXT_COL].tolist(),
        "label": [config.LABEL2ID[label] for label in df[config.LABEL_COL]],
    })
    return data.map(lambda batch: tokenizer(batch["text"], truncation=True,
                                            max_length=config.MAX_TOKENS),
                    batched=True, remove_columns=["text"])


def compute_metrics(eval_pred) -> dict:
    logits, label_ids = eval_pred
    predicted = [config.ID2LABEL[i] for i in np.argmax(logits, axis=-1)]
    actual = [config.ID2LABEL[i] for i in label_ids]
    metrics = utils.binary_metrics(actual, predicted)
    return {k: metrics[k] for k in ("accuracy", "precision_scam", "recall_scam",
                                    "f1_scam", "false_positive_rate")}


def main(smoke: bool = False) -> None:
    utils.set_seed()
    device = utils.get_device()
    limit = config.SMOKE_SAMPLE_SIZE if smoke else None
    out_dir = utils.finetuned_dir(smoke)
    checkpoint_dir = out_dir.parent / f"{config.FINETUNED_DIRNAME}-checkpoints"

    train_df = utils.load_split("train", limit=limit)
    val_df = utils.load_split("validation", limit=limit)
    print(f"Fine-tuning {config.P1_CHOSEN_BASE} on {len(train_df)} messages "
          f"(validation {len(val_df)}) using {utils.device_name(device)}")

    tokenizer = AutoTokenizer.from_pretrained(config.P1_CHOSEN_BASE)
    # Re-label the classification head with our label names; the base model's
    # class 1 is already "spam", which lines up with our class 1 = "scam".
    model = AutoModelForSequenceClassification.from_pretrained(
        config.P1_CHOSEN_BASE, num_labels=len(config.LABEL2ID),
        id2label=config.ID2LABEL, label2id=config.LABEL2ID)

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=1 if smoke else config.TRAIN_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_steps=config.WARMUP_RATIO,     # a float < 1 is read as a ratio of all steps
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_scam",
        fp16=(device == "cuda"),          # mixed precision only on NVIDIA GPUs
        seed=config.SEED,
        logging_steps=50,
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=to_dataset(train_df, tokenizer),
        eval_dataset=to_dataset(val_df, tokenizer),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    try:
        with utils.Timer() as timer:
            trainer.train()
    except RuntimeError as err:
        if "out of memory" in str(err).lower():
            raise SystemExit("GPU ran out of memory - lower TRAIN_BATCH_SIZE in config.py") from err
        raise

    best_scores = trainer.evaluate()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    log = {
        "base_model": config.P1_CHOSEN_BASE,
        "device": utils.device_name(device),
        "train_rows": len(train_df), "validation_rows": len(val_df),
        "hyper_parameters": {k: getattr(args, k) for k in (
            "num_train_epochs", "learning_rate", "per_device_train_batch_size",
            "weight_decay", "warmup_steps", "fp16", "seed")} | {"max_tokens": config.MAX_TOKENS},
        "training_minutes": round(timer.seconds / 60, 2),
        "best_validation_scores": best_scores,
        "epoch_history": [entry for entry in trainer.state.log_history if "eval_loss" in entry],
    }
    utils.save_json(log, utils.results_dir(smoke) / "finetune_log.json")
    print(f"\nSaved fine-tuned model -> {out_dir}")
    print(f"Training time {log['training_minutes']} min; best validation "
          f"F1 {best_scores.get('eval_f1_scam', float('nan')):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the Pipeline 1 model")
    parser.add_argument("--smoke", action="store_true", help="tiny sample, quick code check")
    main(parser.parse_args().smoke)
