from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


def evaluate_classification(y_true, y_pred):
    """Calculate the project's standard classification metrics."""

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "confusion_matrix": cm,
    }
