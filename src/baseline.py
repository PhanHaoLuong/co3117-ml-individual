import numpy as np

from data import load_har_data
from metrics import evaluate_classification


def majority_class(y):
    """Return the most frequent class in the labels."""
    classes, counts = np.unique(y, return_counts=True)
    return classes[np.argmax(counts)]


def predict_majority_class(y, class_label):
    """Predict the same class for every sample."""
    return np.full(len(y), class_label)


def run_baseline():
    data = load_har_data()

    y_train = data["y_train"]
    y_val = data["y_val"]

    majority = majority_class(y_train)
    y_pred = predict_majority_class(y_val, majority)

    results = evaluate_classification(y_val, y_pred)

    print("Majority class:", majority)
    print("Validation Macro-F1:", results["macro_f1"])
    print("Validation Accuracy:", results["accuracy"])
    print("Confusion Matrix:")
    print(results["confusion_matrix"])


if __name__ == "__main__":
    run_baseline()