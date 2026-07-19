"""
Requirement #8 — Fine-tune a model on a Customer Service dataset.

We fine-tune DistilBERT (an SLM, 66M params) on the Bitext Customer Support
dataset (~27k utterances, 27 real support intents such as cancel_order,
get_refund, delivery_period, payment_issue...).

Runs in ~10 minutes on a FREE Google Colab T4 GPU.

In Colab:
    !pip install -q transformers datasets evaluate accelerate scikit-learn
    !python finetune_intent.py
Then download the ./intent-model folder (or zip it) and place it next to
app.py — the Gradio app will automatically load it and compare it live
against the zero-shot baseline.

LLMOps angle: the script prints accuracy + macro-F1 on a held-out test set.
Quote these numbers in your report as the model-quality metric.
"""

import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer,
                          DataCollatorWithPadding)
import evaluate

BASE_MODEL = "distilbert-base-uncased"
DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
OUTPUT_DIR = "./intent-model"
MAX_SAMPLES = 8000          # subsample for fast training; raise for better scores
EPOCHS = 2

# ---------------------------------------------------------------- load data
print("Loading dataset:", DATASET)
ds = load_dataset(DATASET, split="train").shuffle(seed=42)
if MAX_SAMPLES:
    ds = ds.select(range(min(MAX_SAMPLES, len(ds))))

labels = sorted(set(ds["intent"]))
label2id = {l: i for i, l in enumerate(labels)}
id2label = {i: l for l, i in label2id.items()}
print(f"{len(ds)} samples, {len(labels)} intents:", labels)

ds = ds.map(lambda x: {"label": label2id[x["intent"]]})
ds = ds.train_test_split(test_size=0.15, seed=42)

# ---------------------------------------------------------------- tokenize
tok = AutoTokenizer.from_pretrained(BASE_MODEL)


def tokenize(batch):
    return tok(batch["instruction"], truncation=True, max_length=128)


ds = ds.map(tokenize, batched=True)

# ---------------------------------------------------------------- model
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL, num_labels=len(labels),
    id2label=id2label, label2id=label2id,
)

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, y = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=y)["accuracy"],
        "macro_f1": f1.compute(predictions=preds, references=y,
                               average="macro")["f1"],
    }


args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    processing_class=tok,
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=compute_metrics,
)

# ---------------------------------------------------------------- train
trainer.train()

metrics = trainer.evaluate()
print("\n===== FINAL TEST METRICS (quote these in the report) =====")
print(f"Accuracy : {metrics['eval_accuracy']:.4f}")
print(f"Macro F1 : {metrics['eval_macro_f1']:.4f}")

trainer.save_model(OUTPUT_DIR)
tok.save_pretrained(OUTPUT_DIR)
print(f"\nFine-tuned model saved to {OUTPUT_DIR} — copy this folder next to app.py")

# quick sanity check
from transformers import pipeline
pipe = pipeline("text-classification", model=OUTPUT_DIR)
for s in ["I want my money back for order 4521",
          "when will my package arrive?",
          "please cancel my subscription"]:
    print(s, "->", pipe(s)[0])
