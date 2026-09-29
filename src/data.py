from pathlib import Path

import numpy as np


# Path to the extracted UCI HAR Dataset
DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "human+activity+recognition+using+smartphones"
    / "UCI HAR Dataset"
)


# Subjects selected for our validation set
VALIDATION_SUBJECTS = np.array([2, 9, 14, 19])

def load_raw_data():
    """Load the original UCI HAR train and test files."""

    X_train = np.loadtxt(DATA_DIR / "train" / "X_train.txt")
    y_train = np.loadtxt(DATA_DIR / "train" / "y_train.txt", dtype=int)
    subject_train = np.loadtxt(
        DATA_DIR / "train" / "subject_train.txt",
        dtype=int,
    )

    X_test = np.loadtxt(DATA_DIR / "test" / "X_test.txt")
    y_test = np.loadtxt(DATA_DIR / "test" / "y_test.txt", dtype=int)
    subject_test = np.loadtxt(
        DATA_DIR / "test" / "subject_test.txt",
        dtype=int,
    )

    return (
        X_train,
        y_train,
        subject_train,
        X_test,
        y_test,
        subject_test,
    )

def split_train_validation(X, y, subjects):
    """Split the official training data by subject."""

    validation_mask = np.isin(subjects, VALIDATION_SUBJECTS)
    training_mask = ~validation_mask

    X_train = X[training_mask]
    y_train = y[training_mask]

    X_val = X[validation_mask]
    y_val = y[validation_mask]

    return X_train, y_train, X_val, y_val

def load_har_data():
    """Load the HAR dataset using the project's fixed split."""

    (
        X_train_original,
        y_train_original,
        subject_train,
        X_test,
        y_test,
        subject_test,
    ) = load_raw_data()

    X_train, y_train, X_val, y_val = split_train_validation(
        X_train_original,
        y_train_original,
        subject_train,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
