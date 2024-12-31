# import pandas as pd
# import os
#
# # Define paths
# base_path = "/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-HumanT/csv"
# tasks = ["BoolQ", "CB", "COPA", "MultiRC", "ReCoRD", "RTE", "WSC"]
#
# def fix_csv_files(task):
#     # File paths for the task's train, validation, and test CSV files
#     train_file = os.path.join(base_path, task, "train.csv")
#     val_file = os.path.join(base_path, task, "val.csv")
#     test_file = os.path.join(base_path, task, "test.csv")
#
#     # Read the train, val, and test CSV files into DataFrames
#     train_df = pd.read_csv(train_file)
#     val_df = pd.read_csv(val_file)
#     test_df = pd.read_csv(test_file)
#
#     # Ensure the 'test.csv' has a 'label' column (adding dummy values if missing)
#     if 'label' not in test_df.columns:
#         print(f"Adding 'label' column to {task}'s test.csv file.")
#         test_df['label'] = None  # Set 'None' or a placeholder value for testing
#
#     # Rename columns for consistency (if needed)
#     # Rename 'passage' to 'text' and 'question' to 'question_text'
#     train_df = train_df.rename(columns={"passage": "text", "question": "question_text"})
#     val_df = val_df.rename(columns={"passage": "text", "question": "question_text"})
#     test_df = test_df.rename(columns={"passage": "text", "question": "question_text"})
#
#     # Save the fixed CSV files back to disk
#     train_df.to_csv(train_file, index=False)
#     val_df.to_csv(val_file, index=False)
#     test_df.to_csv(test_file, index=False)
#
#     print(f"Fixed CSV files for task: {task}.")
#
# # Loop over all tasks and fix the CSV files
# for task in tasks:
#     fix_csv_files(task)

import pandas as pd
import os

# Define paths
base_path = "/Users/marko/dev/maga/context-debias/data/SloSuperGLUE/SuperGLUE-HumanT/csv"
tasks = ["BoolQ", "CB", "COPA", "MultiRC", "ReCoRD", "RTE", "WSC"]

def undo_csv_fixes(task):
    # File paths for the task's train, validation, and test CSV files
    train_file = os.path.join(base_path, task, "train.csv")
    val_file = os.path.join(base_path, task, "val.csv")
    test_file = os.path.join(base_path, task, "test.csv")

    # Read the train, val, and test CSV files into DataFrames
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    # Restore the original column names (passage -> passage, question -> question)
    train_df = train_df.rename(columns={"text": "passage", "question_text": "question"})
    val_df = val_df.rename(columns={"text": "passage", "question_text": "question"})
    test_df = test_df.rename(columns={"text": "passage", "question_text": "question"})

    # Remove the 'label' column from the test.csv if it was added
    if 'label' in test_df.columns:
        print(f"Removing 'label' column from {task}'s test.csv file.")
        test_df = test_df.drop(columns=["label"])

    # Save the restored CSV files back to disk
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"Restored original CSV files for task: {task}.")

# Loop over all tasks and undo the fixes
for task in tasks:
    undo_csv_fixes(task)

