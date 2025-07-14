# Imports
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import evaluate
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    TFAutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    create_optimizer,
)

# Configuration
CONFIG = {
    "checkpoint": "google-t5/t5-small",
    "batch_size": 8,
    "max_input_length": 512,
    "max_target_length": 256,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "epochs": 5,
    "warmup_ratio": 0.1,
    "output_dir": "saved_models/hieroglyphic_to_english",
    "prefix": "translate hieroglyphic to english: ",
    "seed": 42,
}

# Setup
os.environ['PYTHONHASHSEED'] = str(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
tf.random.set_seed(CONFIG["seed"])

# GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load Dataset
df = pd.read_csv("PREeprocessed_dataset.csv").dropna(subset=["source_cleaned", "target_cleaned"])
df = df.rename(columns={"source_cleaned": "input_text", "target_cleaned": "target_text"})
dataset = Dataset.from_pandas(df)

# Split Dataset
train_test = dataset.train_test_split(test_size=0.2, seed=CONFIG["seed"])
valid_test = train_test["test"].train_test_split(test_size=0.5, seed=CONFIG["seed"])
dataset = {
    "train": train_test["train"],
    "validation": valid_test["train"],
    "test": valid_test["test"],
}

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(CONFIG["checkpoint"])

# Preprocessing function
def preprocess_function(examples):
    inputs = [CONFIG["prefix"] + text for text in examples["input_text"]]
    model_inputs = tokenizer(inputs, max_length=CONFIG["max_input_length"], truncation=True, padding="max_length")
    labels = tokenizer(examples["target_text"], max_length=CONFIG["max_target_length"], truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize datasets
tokenized_datasets = {
    split: ds.map(preprocess_function, batched=True, remove_columns=ds.column_names)
    for split, ds in dataset.items()
}

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=CONFIG["checkpoint"], return_tensors="tf")

# Prepare TensorFlow datasets
tf_train_set = TFAutoModelForSeq2SeqLM.from_pretrained(CONFIG["checkpoint"]).prepare_tf_dataset(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size=CONFIG["batch_size"],
    collate_fn=data_collator,
)

tf_val_set = TFAutoModelForSeq2SeqLM.from_pretrained(CONFIG["checkpoint"]).prepare_tf_dataset(
    tokenized_datasets["validation"],
    shuffle=False,
    batch_size=CONFIG["batch_size"],
    collate_fn=data_collator,
)

tf_test_set = TFAutoModelForSeq2SeqLM.from_pretrained(CONFIG["checkpoint"]).prepare_tf_dataset(
    tokenized_datasets["test"],
    shuffle=False,
    batch_size=CONFIG["batch_size"],
    collate_fn=data_collator,
)

# Optimizer and Scheduler
train_steps = (tokenized_datasets["train"].num_rows // CONFIG["batch_size"]) * CONFIG["epochs"]
warmup_steps = int(CONFIG["warmup_ratio"] * train_steps)

optimizer, lr_schedule = create_optimizer(
    init_lr=CONFIG["learning_rate"],
    num_train_steps=train_steps,
    num_warmup_steps=warmup_steps,
    weight_decay_rate=CONFIG["weight_decay"],
)

# Load Model
# model = TFAutoModelForSeq2SeqLM.from_pretrained(CONFIG["output_dir"])
model = TFAutoModelForSeq2SeqLM.from_pretrained(CONFIG["checkpoint"])
model.compile(optimizer=optimizer)

# Train Model
history = model.fit(
    tf_train_set,
    validation_data=tf_val_set,
    epochs=CONFIG["epochs"],
)

# Save Model
model.save_pretrained(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])

# Evaluation
metric_bleu = evaluate.load("sacrebleu")
metric_rouge = evaluate.load("rouge")

def compute_metrics(tf_dataset):
    preds, labels = [], []

    for batch_input, batch_labels in tqdm(tf_dataset):
        input_ids = batch_input["input_ids"]
        attention_mask = batch_input.get("attention_mask", None)

        # Generate predictions
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=CONFIG["max_target_length"]
        )

        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(output, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(batch_labels.numpy(), skip_special_tokens=True)

        preds.extend(decoded_preds)
        labels.extend(decoded_labels)

    # Print sample
    for i in range(3):
        print(f"\nExample {i + 1}")
        print("Prediction: ", preds[i])
        print("Reference:  ", labels[i])

    # Compute BLEU and ROUGE
    bleu = metric_bleu.compute(predictions=preds, references=[[l] for l in labels])
    rouge = metric_rouge.compute(predictions=preds, references=labels)

    return {
        "BLEU": bleu["score"],
        "ROUGE-1": rouge["rouge1"],
        "ROUGE-2": rouge["rouge2"],
        "ROUGE-L": rouge["rougeL"]
    }

# Run Evaluation
results = compute_metrics(tf_test_set)
print("Test Results:", results)



