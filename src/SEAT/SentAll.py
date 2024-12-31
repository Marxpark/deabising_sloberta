import re

import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer, Trainer, AutoModelForSequenceClassification, TrainingArguments

if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.empty_cache()
else:
    device = 'cpu'


def extract_path(full_path):
    # Use regex to extract the value between "param_search/" and "/checkpoint-best"
    match = re.search(r'/param_search/(.*?)/checkpoint-best', full_path)
    if match:
        return match.group(1)  # The captured path part
    return "SLOBERTAORIGINAL"  # If no match is found


from datasets import load_dataset, ClassLabel
from evaluate import load

dataset = load_dataset("cjvt/sentinews", 'sentence_level')

print(dataset)
# Load metrics
accuracy_metric = load("accuracy")
f1_metric = load("f1")


# Custom Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = logits.argmax(axis=-1)

    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_negative = f1_score(labels, predictions, average=None, zero_division=0)[0]
    f1_neutral = f1_score(labels, predictions, average=None, zero_division=0)[1]
    f1_positive = f1_score(labels, predictions, average=None, zero_division=0)[2]

    precision_negative = precision_score(labels, predictions, average=None, zero_division=0)[0]
    precision_neutral = precision_score(labels, predictions, average=None, zero_division=0)[1]
    precision_positive = precision_score(labels, predictions, average=None, zero_division=0)[2]

    recall_negative = recall_score(labels, predictions, average=None, zero_division=0)[0]
    recall_neutral = recall_score(labels, predictions, average=None, zero_division=0)[1]
    recall_positive = recall_score(labels, predictions, average=None, zero_division=0)[2]

    return {
        "eval_accuracy": (predictions == labels).mean(),
        "eval_f1_macro": f1_macro,
        "eval_f1_negative": f1_negative,
        "eval_f1_neutral": f1_neutral,
        "eval_f1_positive": f1_positive,
        "eval_precision_negative": precision_negative,
        "eval_precision_neutral": precision_neutral,
        "eval_precision_positive": precision_positive,
        "eval_recall_negative": recall_negative,
        "eval_recall_neutral": recall_neutral,
        "eval_recall_positive": recall_positive,
    }


def run_senti_analaysis(path_to_model="EMBEDDIA/sloberta", save_path="SLOBERTA"):
    tokenizer = AutoTokenizer.from_pretrained(path_to_model)

    # model = AutoModel.from_pretrained(path_to_model)
    num_labels = 3
    model = AutoModelForSequenceClassification.from_pretrained(path_to_model, num_labels=num_labels)
    model = model.to(device)

    label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    class_label = ClassLabel(num_classes=len(label_mapping), names=list(label_mapping.keys()))
    casted_set = dataset.cast_column("sentiment", class_label)

    train_val_test_split = casted_set["train"].train_test_split(train_size=0.9, test_size=0.1, seed=42)
    train_dataset_raw = train_val_test_split["train"]
    val_test_split = train_val_test_split["test"].train_test_split(train_size=0.5, test_size=0.5, seed=42)
    val_dataset_raw = val_test_split["train"]
    test_dataset_raw = val_test_split["test"]

    def preprocess_function(examples):
        tokenized_data = tokenizer(
            examples["content"],
            truncation=True,
            padding="max_length",
            max_length=79,
        )
        tokenized_data["label"] = examples["sentiment"]  # Already numeric
        return tokenized_data

    # Apply tokenization
    train_dataset = train_dataset_raw.map(preprocess_function, batched=True)
    val_dataset = val_dataset_raw.map(preprocess_function, batched=True)
    test_dataset = test_dataset_raw.map(preprocess_function, batched=True)
    # Set the dataset format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir=f"./{save_path}-results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=36,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        logging_dir='./logs'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    evaluation_results = trainer.evaluate(test_dataset)

    structured_report = {
        "sentiment": {
            "loss": evaluation_results["eval_loss"],
            "metrics": {
                "accuracy": evaluation_results["eval_accuracy"],
                "f1_macro": evaluation_results["eval_f1_macro"],
                "f1_negative": evaluation_results["eval_f1_negative"],
                "f1_neutral": evaluation_results["eval_f1_neutral"],
                "f1_positive": evaluation_results["eval_f1_positive"],
                "precision_negative": evaluation_results["eval_precision_negative"],
                "precision_neutral": evaluation_results["eval_precision_neutral"],
                "precision_positive": evaluation_results["eval_precision_positive"],
                "recall_negative": evaluation_results["eval_recall_negative"],
                "recall_neutral": evaluation_results["eval_recall_neutral"],
                "recall_positive": evaluation_results["eval_recall_positive"],
            }
        }
    }

    # Save predictions
    predictions = trainer.predict(test_dataset)
    # Safely handle logits
    logits = predictions.predictions
    if isinstance(logits, tuple):
        logits = logits[0]  # Extract the logits if it's a tuple

    pred_labels = logits.argmax(axis=-1)

    # Combine inputs, predictions, and true labels for saving
    examples = test_dataset_raw.to_pandas()
    examples["predicted_label"] = pred_labels
    examples["true_label"] = examples["sentiment"].map(label_mapping)

    examples.to_csv(f"{save_path}_sent_predictions.csv", index=False)
    return structured_report


def compare_predictions(model1_path, model2_path, output_path):
    """
    Compare predictions between two models.

    Args:
        model1_path (str): Path to predictions CSV of the first model.
        model2_path (str): Path to predictions CSV of the second model.
        output_path (str): Path to save comparison results.
    """
    import pandas as pd

    # Load predictions
    model1_preds = pd.read_csv(model1_path)
    model2_preds = pd.read_csv(model2_path)

    # Compare predictions
    comparison = model1_preds.copy()
    comparison["model2_predicted_label"] = model2_preds["predicted_label"]
    comparison["difference"] = comparison["predicted_label"] != comparison["model2_predicted_label"]

    # Save differences to file
    differences = comparison[comparison["difference"]]
    differences.to_csv(output_path, index=False)
    print(f"Saved prediction differences to {output_path}")
