import json
from collections import defaultdict

import torch

if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.empty_cache()
else:
    device = 'cpu'


import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

batch_size = 8
num_epochs = 3
train_test_split_pct = 0.05

def run_boolq(model_name, tokenizer, base_path):
    # Define the compute_metrics function
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = model.to(device)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)  # Get the predicted class (0 or 1)
        accuracy = np.mean(predictions == labels)  # Compute accuracy
        return {"accuracy": accuracy}

    dataset = load_dataset("json", data_files={
        "validation": base_path + "BoolQ/val.jsonl",
        "train": base_path + "BoolQ/train.jsonl"
    })

    def remap_labels(example):
        return {"label_int": int(example["label"])}  # Debugging: log updated label

    dataset = dataset.map(remap_labels, load_from_cache_file=False)
    dataset = dataset.remove_columns("label").rename_column("label_int", "label")

    #####
    # Preprocess the dataset
    def preprocess_function(examples):
        return tokenizer(
            examples["passage"],
            examples["question"],
            truncation=True,
            padding="max_length",
            max_length=512
        )


    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    tokenized_datasets.remove_columns(['idx', 'passage', 'question'])
    tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    split_dataset = tokenized_datasets["train"].train_test_split(test_size=train_test_split_pct, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    val_dataset = tokenized_datasets["validation"]


    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()
    eval = trainer.evaluate(test_dataset)
    del model
    return eval


def run_cb(model_name, tokenizer, base_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model = model.to(device)
    dataset = load_dataset("json", data_files={
        "validation": base_path + "CB/val.jsonl",
        "train": base_path + "CB/train.jsonl"
    })

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = logits.argmax(axis=-1)  # Get the predicted class
        accuracy = accuracy_score(labels, predictions)  # Accuracy
        f1 = f1_score(labels, predictions, average="weighted")  # Weighted F1-score
        return {
            "accuracy": accuracy,
            "f1": f1,
        }

    label_mapping = {"contradiction": 0, "neutral": 1, "entailment": 2}
    def remap_labels(example):
        example["label"] = label_mapping[example["label"]]
        return example

    dataset = dataset.map(remap_labels, load_from_cache_file=False)

    def preprocess_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    columns_to_remove = ["premise", "hypothesis", "idx"]
    tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    split_dataset = tokenized_dataset["train"].train_test_split(test_size=train_test_split_pct, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    val_dataset = tokenized_dataset["validation"]

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_results = trainer.evaluate(test_dataset)
    del model
    return test_results


def run_copa(model_name, tokenizer, base_path, save_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = model.to(device)
    dataset = load_dataset("json", data_files={
        "validation": base_path + "COPA/val.jsonl",
        "train": base_path + "COPA/train.jsonl"
    })

    def compute_metrics(eval_pred):
        """
        Compute accuracy for COPA based on logits and labels.
        Consecutive entries correspond to premise-choice1 and premise-choice2.
        """
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)

        # Group logits and labels by example_id
        grouped_predictions = defaultdict(list)
        grouped_labels = defaultdict(list)

        for i, (pred, label) in enumerate(zip(predictions, labels)):
            example_id = i // 2  # Group every two consecutive entries by example_id
            grouped_predictions[example_id].append(pred)
            grouped_labels[example_id].append(label)

        # Evaluate grouped predictions
        correct = 0
        total = 0
        for example_id, preds in grouped_predictions.items():
            if len(preds) == 2:  # Ensure both choices are present
                logits_choice1, logits_choice2 = logits[example_id * 2], logits[example_id * 2 + 1]
                chosen = np.argmax([logits_choice1, logits_choice2])
                label_choice = 0 if grouped_labels[example_id][0] == 0 else 1  # Adjust based on label format
                if chosen == label_choice:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0
        return {"accuracy": accuracy}

    question_translation = {
        "cause": "Kaj je vzrok za to?",
        "effect": "Kaj je učinek tega?"
    }

    def create_reformatted_dataset(dataset):
        """
        Reformat the COPA dataset to create inputs for both choices, with example IDs.
        """
        reformatted_splits = {}

        for split in dataset.keys():
            examples = dataset[split]
            new_data = {
                "inputs": [],
                "labels": [],
                "example_id": [],
            }

            for idx, (premise, question, choice1, choice2, label) in enumerate(
                    zip(
                        examples["premise"],
                        examples["question"],
                        examples["choice1"],
                        examples["choice2"],
                        examples["label"],
                    )
            ):
                # Add choice1
                new_data["inputs"].append(f"{premise} {question_translation[question]} {choice1}")
                new_data["labels"].append(label)  # Binary label for choice1
                new_data["example_id"].append(idx)

                # Add choice2
                new_data["inputs"].append(f"{premise} {question_translation[question]} {choice2}")
                new_data["labels"].append(label)  # Binary label for choice2
                new_data["example_id"].append(idx)

            reformatted_splits[split] = Dataset.from_dict(new_data)

        return DatasetDict(reformatted_splits)

    def preprocess_function(examples):
        return tokenizer(
            examples["inputs"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    reformatted_splits = create_reformatted_dataset(dataset)
    reformatted_splits = reformatted_splits.map(preprocess_function, batched=True)

    # Remove the old "label" column if necessary
    reformatted_splits.set_format(
        type="torch",
        columns=[
            "inputs",
            "labels",
            "example_id",
            "input_ids",
            "attention_mask",

        ],
    )

    split_dataset = reformatted_splits["train"].train_test_split(test_size=train_test_split_pct, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=reformatted_splits["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_results = trainer.evaluate(test_dataset)
    del model
    return test_results


def run_rte(model_name, tokenizer, base_path, save_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary classification
    model = model.to(device)
    dataset = load_dataset("json", data_files={
        "validation": base_path + "RTE/val.jsonl",
        "train": base_path + "RTE/train.jsonl"
    })

    # Map labels to integers
    def map_labels(example):
        example['label'] = 0 if example['label'] == 'entailment' else 1
        return example

    # Apply mapping to the dataset
    dataset = dataset.map(map_labels)

    # Define compute_metrics function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)  # Get the predicted class (0 or 1)
        accuracy = np.mean(predictions == labels)  # Compute accuracy
        return {"accuracy": accuracy}

    def preprocess_function(example):
        return tokenizer(
            example["premise"],
            example["hypothesis"],
            padding="max_length",
            truncation=True,
            max_length=256
        )

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    split_dataset = tokenized_dataset["train"].train_test_split(test_size=train_test_split_pct, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    eval_dataset = tokenized_dataset["validation"]  # Adjust to match your dataset splits

    training_args = TrainingArguments(
        output_dir="./results",  # Output directory
        evaluation_strategy="epoch",  # Evaluate each epoch
        save_strategy="epoch",  # Save model each epoch
        learning_rate=2e-5,  # Learning rate
        per_device_train_batch_size=batch_size,  # Batch size
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,  # Number of epochs
        weight_decay=0.01,  # Weight decay
        logging_dir='./logs',  # Directory for logs
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    results = trainer.evaluate(test_dataset)

    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predicted_labels = np.argmax(logits, axis=-1)
    true_labels = test_dataset["label"]

    # Save predictions to save_path
    with open(save_path, "w") as f:
        for idx, (true_label, predicted_label) in enumerate(zip(true_labels, predicted_labels)):
            f.write(f"{idx}\t{true_label}\t{predicted_label}\n")

    print(results)
    del model
    return results


def process_multirc_file(file_path):
    rows = []
    # Load and preprocess
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            passage_text = data["passage"]["text"]
            passage_idx = data["idx"]
            questions = data["passage"]["questions"]

            for question_id, question_data in questions.items():
                question_text = question_data.get("question")
                if not question_text:  # Skip if 'question' is None or empty
                    continue
                question_idx = question_data["idx"]
                answers = question_data["answers"]

                for answer_id, answer_data in answers.items():
                    answer_text = answer_data.get("text")
                    if not answer_text:  # Skip if 'text' is None or empty
                        continue
                    answer_label = answer_data["label"]
                    answer_idx = answer_data["idx"]

                    rows.append({
                        "passage_text": passage_text,
                        "passage_idx": passage_idx,
                        "question_id": int(question_id),
                        "question_text": question_text,
                        "question_idx": int(question_idx),
                        "answer_id": int(answer_id),
                        "answer_text": answer_text,
                        "answer_label": int(float(answer_label)),
                        "answer_idx": int(answer_idx)
                    })
    return rows


def run_multirc(model_name, tokenizer, base_path, save_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary classification
    model = model.to(device)

    train_file = process_multirc_file(base_path + "MultiRC/train.jsonl")
    val_file = process_multirc_file(base_path + "MultiRC/val.jsonl")

    train_dataset = Dataset.from_list(train_file)
    val_dataset = Dataset.from_list(train_file)

    print(train_dataset)

model_name = "EMBEDDIA/sloberta"
base_path="/Users/marko/dev/maga/context-debias/data/tasks/humanTranslation/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bool_q_res = run_boolq(model_name, tokenizer, base_path)
cb_res = run_cb(model_name, tokenizer, base_path)
run_copa(model_name, tokenizer, base_path, "")

def run_tasks(model_name_or_path="EMBEDDIA/sloberta", base_path="/Users/marko/dev/maga/context-debias/data/tasks/humanTranslation/", save_path=""):
    model_name = model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bool_q_res = run_boolq(model_name, tokenizer, base_path)
    cb_res = run_cb(model_name, tokenizer, base_path)
    copa_res = run_copa(model_name, tokenizer, base_path, save_path)
    rte_res = run_rte(model_name, tokenizer, base_path, save_path)
    return {
        # "boolq": bool_q_res,
        # "cb": cb_res,
        "copa": copa_res,
        # "rte": rte_res
    }