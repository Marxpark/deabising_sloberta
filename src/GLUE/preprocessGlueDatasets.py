# i have this kind of file structure
# /Users/marko/dev/maga/context-debias/data/SloSuperGLUE/
#
# here there are 3 directories
# [
# 'SuperGLUE-GoogleMT',
# 'SuperGLUE-HumanT',
# 'SuperGLUE-HumanTmini'
# ]
#
# in each of these directories is a directory called 'csv'
#
# in each of the csv directories there are the following directories
# [
# 'BoolQ'
# 'CB'
# 'COPA'
# 'MultiRC'
# 'ReCoRD'
# 'RTE'
# 'WSC'
# ]
#
# in each of those directories, there are two csv files containing the train and validation sets for the tasks called
#
#     'train.csv' and 'val.csv'
#
# in directories 'SuperGLUE-GoogleMT','SuperGLUE-HumanT', i want you to move a random 10% of the rows from each train csv and into a new file in that directory called test.csv. the rows selected and moved should be removed from the train file.
#
#
# in directory 'SuperGLUE-HumanTmini' i want you to chorten all train files to 10% of what they are currently

import os
import pandas as pd
from pathlib import Path
import random

# Set up the paths
base_path = Path("/Users/marko/dev/maga/context-debias/data/SloSuperGLUE")
directories = [
    "SuperGLUE-GoogleMT",
    "SuperGLUE-HumanT",
    "SuperGLUE-HumanTmini"
]

# Process SuperGLUE-GoogleMT and SuperGLUE-HumanT for test set extraction
for dir_name in directories[:2]:  # Only the first two directories
    dir_path = base_path / dir_name / "csv"
    for task_dir in dir_path.iterdir():
        if task_dir.is_dir():
            train_file = task_dir / "train.csv"
            val_file = task_dir / "val.csv"
            test_file = task_dir / "test.csv"

            # Load the train data
            train_data = pd.read_csv(train_file)
            test_size = int(0.1 * len(train_data))

            # Randomly sample 10% of rows for the test set
            test_data = train_data.sample(n=test_size, random_state=42)
            train_data = train_data.drop(test_data.index)

            # Save the new train and test sets
            train_data.to_csv(train_file, index=False)
            test_data.to_csv(test_file, index=False)

# Process SuperGLUE-HumanTmini to shorten train sets to 10% of current size
humanTmini_path = base_path / "SuperGLUE-HumanTmini" / "csv"
for task_dir in humanTmini_path.iterdir():
    if task_dir.is_dir():
        train_file = task_dir / "train.csv"

        # Load the train data
        train_data = pd.read_csv(train_file)
        mini_size = int(0.1 * len(train_data))

        # Randomly sample 10% of rows to keep
        mini_train_data = train_data.sample(n=mini_size, random_state=42)

        # Save the reduced train set
        mini_train_data.to_csv(train_file, index=False)

print("Processing complete.")