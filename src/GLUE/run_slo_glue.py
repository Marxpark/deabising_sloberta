import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_dataset, Dataset
import evaluate
import pandas as pd
from torch import nn
# Initialize the metric with the evaluate library
metric = evaluate.load("accuracy")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
# Define paths
# base_path = "/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-HumanTmini/csv"
tasks = ["BoolQ", "CB", "COPA", "RTE"]
# tasks = ["MultiRC", "ReCoRD", "RTE", "WSC"]
if torch.cuda.is_available():
    device = 'cuda'
elif torch.has_mps:
    device = 'mps'
else:
    device = 'cpu'

results_dict = {}


# Function to parse nested column names
def parse_nested_column(col_name):
    matches = re.findall(r'([a-zA-Z]+)(?:\.(\d+))?', col_name)
    structure = []
    for match in matches:
        if match[1]:  # Indexed field
            structure.append((match[0], int(match[1])))
        else:  # Non-indexed field
            structure.append((match[0], None))
    return structure


def process_multirc_task(path, task):
    # Load data files for MultiRC task
    data_files = {
        "train": os.path.join(path, task, "train.csv"),
        "validation": os.path.join(path, task, "val.csv"),
        "test": os.path.join(path, task, "test.csv"),
    }
    datasets = {}

    for split, file_path in data_files.items():
        df = pd.read_csv(file_path, dtype=str).fillna("")  # Load as strings, fill NAs with empty strings
        expanded_data = []

        def process_multirc_row(row):
            # Initialize row dictionary to hold parsed data
            row_data = {}

            # Flatten row data according to nested structure
            for col_name, value in row.items():
                structure = parse_nested_column(col_name)
                temp = row_data
                for level, (key, idx) in enumerate(structure):
                    if idx is not None:
                        if key not in temp:
                            temp[key] = []
                        while len(temp[key]) <= idx:
                            temp[key].append({})
                        temp = temp[key][idx]
                    else:
                        if level == len(structure) - 1:
                            temp[key] = value
                        else:
                            if key not in temp:
                                temp[key] = {}
                            temp = temp[key]

            # Extract passage text
            passage_text = row_data.get("passage", {}).get("text", "")

            passage_text = row_data.get("passage", {}).get("text", "")
            for question in row_data.get("passage", {}).get("questions", []):
                question_text = question.get("question", "")
                correct_answers = [ans.get("text", "") for ans in question.get("answers", []) if
                                   ans.get("label") == "1"]

                if correct_answers:
                    aggregated_answer = " ".join(correct_answers)
                    expanded_data.append({
                        "passage": passage_text,
                        "question": question_text,
                        "answer": aggregated_answer,
                        "label": 1  # Correct answer indicator
                    })

        # Process each row in the DataFrame to extract questions with aggregated answers
        df.apply(process_multirc_row, axis=1)

        # Convert processed data to Hugging Face Dataset
        datasets[split] = Dataset.from_pandas(pd.DataFrame(expanded_data))

    return datasets


# Helper function for loading datasets from CSV files
def load_task_data(task, path):
    if task == "MultiRC":
        return process_multirc_task(path, task)
    elif task == "ReCoRD":
        # Load ReCoRD with pandas for column consistency
        data_files = {
            "train": os.path.join(path, task, "train.csv"),
            "validation": os.path.join(path, task, "val.csv"),
            "test": os.path.join(path, task, "test.csv"),
        }
        datasets = {}
        for split, file_path in data_files.items():
            df = pd.read_csv(file_path, dtype=str).fillna("")  # Fill missing values with empty strings

            # Define a function to extract labels based on answer columns and return integers
            def extract_labels(row):
                correct_answers = []
                for col in row.index:
                    if "answers" in col and "text" in col:
                        correct_answers.append(row[col])
                # Map each answer to an integer label, e.g., use hash or ordinal mapping
                # Here, using `0` as a placeholder; adjust based on actual answer matching
                return 0 if correct_answers else 1

            # Convert the label to an integer
            df["label"] = df.apply(extract_labels, axis=1).astype(int)
            datasets[split] = Dataset.from_pandas(df)
        return datasets
    elif task == "RTE":
        # Load RTE with pandas for column consistency
        data_files = {
            "train": os.path.join(path, task, "train.csv"),
            "validation": os.path.join(path, task, "val.csv"),
            "test": os.path.join(path, task, "test.csv"),
        }
        datasets = {}
        for split, file_path in data_files.items():
            df = pd.read_csv(file_path, dtype=str).fillna("")  # Load as strings, fill NAs with empty strings

            # Map "entailment" and "not_entailment" to integer labels 1 and 0
            label_mapping = {"entailment": 1, "not_entailment": 0}
            df["label"] = df["label"].map(label_mapping).astype(int)  # Convert labels to integers

            # Convert to Hugging Face Dataset
            datasets[split] = Dataset.from_pandas(df)
        return datasets
    else:
        dataset = load_dataset(
            "csv",
            data_files={
                "train": os.path.join(path, task, "train.csv"),
                "validation": os.path.join(path, task, "val.csv"),
                "test": os.path.join(path, task, "test.csv"),
            }
        )

    # Add label preprocessing if missing (dummy label for test set)
    def add_label_column(examples):
        # For tasks like BoolQ, CB, etc., we add binary labels or task-specific logic
        if task == "BoolQ":
            examples["label"] = [1 if i % 2 == 0 else 0 for i in range(len(examples["passage"]))]
        elif task == "CB":
            examples["label"] = [1 if i % 2 == 0 else 0 for i in range(len(examples["hypothesis"]))]
        elif task == "COPA":
            examples["label"] = [1 if i % 2 == 0 else 0 for i in range(len(examples["premise"]))]
        elif task == "MultiRC":
            examples["label"] = [1 if "correct" in text else 0 for text in
                                 examples["passage.questions.0.answers.0.label"]]

            # Add additional task-specific logic here
        return examples

    # Apply the label column addition
    dataset = dataset.map(add_label_column, batched=True)

    # Rename columns for consistency (passage -> text, question -> question_text)
    if task == "BoolQ":
        dataset = dataset.rename_column("passage", "text")
        dataset = dataset.rename_column("question", "question_text")
    if task == "CB":
        dataset = dataset.rename_column("premise", "text")
        dataset = dataset.rename_column("hypothesis", "question_text")
    elif task == "COPA":
        dataset = dataset.rename_column("premise", "text")
        dataset = dataset.rename_column("question", "question_text")
    elif task == "MultiRC":
        # For MultiRC, map answer columns to ensure compatibility
        dataset = dataset.rename_column("passage.text", "text")
        dataset = dataset.rename_column("passage.questions.0.question", "question_text")


    return dataset

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # Accept additional keyword arguments
        labels = inputs.pop("labels").to(torch.long)
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss()
        #loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Evaluation function to compute accuracy
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    return metric.compute(predictions=preds, references=labels)

def test_model(path_to_model, path, results_file):
    # Train and evaluate on each task
    for task in tasks:
        print(f"\nRunning task: {task}")
        # Initialize model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(path_to_model)
        model = AutoModelForSequenceClassification.from_pretrained(path_to_model, num_labels=2)
        model.to(device)
        data_collator = DataCollatorWithPadding(tokenizer, padding="max_length")
        # Load dataset for the current task
        dataset = load_task_data(task, path)

        zhetoni = tokenizer.tokenize("Transformerji so močni jezikovni modeli!")
        test = tokenizer("Transformerji so močni jezikovni modeli!")

        print(test)

        # Tokenize data
        def preprocess_function(examples):
            if task == "MultiRC":
                # Get passages, questions, and answers
                passages = examples["passage"]
                questions = examples["question"]
                answers = examples["answer"]

                # Tokenize passages, questions, and answers
                tokenized_passages = tokenizer(passages, truncation=True, padding="max_length", max_length=256)
                tokenized_questions = tokenizer(questions, truncation=True, padding="max_length", max_length=128)
                tokenized_answers = tokenizer(answers, truncation=True, padding="max_length", max_length=128)

                # Concatenate tokenized inputs: passage + question + answer
                input_ids = [
                    passage + question + answer for passage, question, answer in
                    zip(tokenized_passages["input_ids"], tokenized_questions["input_ids"],
                        tokenized_answers["input_ids"])
                ]
                attention_masks = [
                    passage + question + answer for passage, question, answer in
                    zip(tokenized_passages["attention_mask"], tokenized_questions["attention_mask"],
                        tokenized_answers["attention_mask"])
                ]

                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_masks,
                    "label": examples["label"],
                }
            elif task == "ReCoRD":
                # For ReCoRD, combine passage text and query for tokenization
                # For ReCoRD, combine passage text and query for tokenization
                texts = [f"{passage} {query}" for passage, query in
                         zip(examples["passage.text"], examples.get("qas.0.query", ""))]
                return tokenizer(texts, truncation=True, padding="max_length", max_length=512)
            elif task == "RTE":
                # For RTE, tokenize the premise and hypothesis as input pairs
                return tokenizer(examples["premise"], examples["hypothesis"], truncation=True, padding="max_length", max_length=512)
            else:
                return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)


        # Apply preprocessing to each dataset split
        if isinstance(dataset, dict):
            # If dataset is a dictionary (e.g., for ReCoRD), apply preprocessing to each split
            encoded_dataset = {}
            for split in dataset:
                if task == "MultiRC":
                    # For MultiRC, preprocess fields individually
                    encoded_dataset[split] = dataset[split].map(preprocess_function, batched=True)
                    # Convert labels to integer
                    encoded_dataset[split] = encoded_dataset[split].map(lambda x: {"label": int(x["label"])})
                else:
                    # Standard preprocessing for other tasks
                    encoded_dataset[split] = dataset[split].map(preprocess_function, batched=True)
        else:
            # For single Dataset objects, apply preprocessing directly
            encoded_dataset = dataset.map(preprocess_function, batched=True)
            encoded_dataset = encoded_dataset.map(lambda x: {"label": int(x["label"])})


        if (device == 'mps'):
            # Define training arguments
            training_args = TrainingArguments(
                output_dir=f"./results/{task}",
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                num_train_epochs=4,
                weight_decay=0.01,
                logging_dir='./logs',
                use_mps_device=True
            )
        else:
            training_args = TrainingArguments(
                output_dir=f"./results/{task}",
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                num_train_epochs=4,
                weight_decay=0.01,
                logging_dir='./logs',
            )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validation"],
            compute_metrics=compute_metrics,
        )

        # Train and evaluate
        trainer.train()
        val_results = trainer.evaluate()  # Evaluate on the validation set instead of test
        results_dict[task] = {
            "accuracy": val_results["eval_accuracy"],
            "loss": val_results["eval_loss"]
        }

        # Write validation results to file
        with open(results_file, "a") as f:
            f.write(f"\nValidation results for {task}:\n")
            f.write(f"Accuracy: {val_results['eval_accuracy']:.2f}, Loss: {val_results['eval_loss']:.4f}\n")

        # Evaluate on the test set if it exists
        if "test" in encoded_dataset:
            test_results = trainer.evaluate(encoded_dataset["test"])
            results_dict[task]["test_accuracy"] = test_results["eval_accuracy"]
            results_dict[task]["test_loss"] = test_results["eval_loss"]

            # Write test results to file
            with open(results_file, "a") as f:
                f.write(f"Test results for {task}:\n")
                f.write(f"Accuracy: {test_results['eval_accuracy']:.2f}, Loss: {test_results['eval_loss']:.4f}\n")

    print("\nBenchmark Summary:")
    for task, metrics in results_dict.items():
        print(f"Task: {task}, Accuracy: {metrics['accuracy']:.2f}, Loss: {metrics['loss']:.4f}")

