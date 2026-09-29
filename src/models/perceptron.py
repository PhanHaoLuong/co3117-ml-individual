import numpy as np

from src.data import load_har_data
from src.metrics import evaluate_classification

rng = np.random.default_rng(42)

def train(x_train, y_train):
    #Original weights & bias
    weights = np.zeros((6, x_train.shape[1]))
    bias = np.zeros(6)

    #Training loop
    n = 0.1 #Learning rate
    for epoch in range(60):
        # indices = [i for i in range(x_train.shape[0])]
        # rng.shuffle(indices)
        for i in range(x_train.shape[0]):
            max_score = -np.inf
            max_index = -1
            for j in range(0, 6):
                score = np.dot(weights[j], x_train[i]) + bias[j]
                if score > max_score:
                    max_score = score
                    max_index = j

            if max_index != (y_train[i] - 1):
                weights[max_index] -= n * x_train[i]
                weights[y_train[i] - 1] += n * x_train[i]
                bias[max_index] -= n
                bias[y_train[i] - 1] += n

    return weights, bias

def predict(x, weights, bias):
    predictions = []
    for i in range(x.shape[0]):
        max_score = -np.inf
        max_index = -1
        for j in range(0, 6):
            score = np.dot(weights[j], x[i]) + bias[j]
            if score > max_score:
                max_score = score
                max_index = j
        predictions.append(max_index + 1)
    return predictions

if __name__ == "__main__":
    data = load_har_data()
    weights, bias = train(data["X_train"], data["y_train"])
    training_results = evaluate_classification(data["y_train"], predict(data["X_train"], weights, bias))
    print("Training Macro-F1:", training_results["macro_f1"])
    print("Training Accuracy:", training_results["accuracy"])
    print("Training Confusion Matrix:")
    print(training_results["confusion_matrix"])

    y_pred = predict(data["X_val"], weights, bias)
    y_val = data["y_val"]

    results = evaluate_classification(y_val, y_pred)
    
    print("Validation Macro-F1:", results["macro_f1"])
    print("Validation Accuracy:", results["accuracy"])
    print("Confusion Matrix:")
    print(results["confusion_matrix"])