from flask import Flask, jsonify, render_template
import numpy as np

from data_pipeline import load_dataset
from neural_network import NeuralNetwork

app = Flask(__name__)


def metrics(y_true, y_pred):
    true, pred = y_true.ravel().astype(int), y_pred.ravel().astype(int)
    tp = int(((true == 1) & (pred == 1)).sum())
    tn = int(((true == 0) & (pred == 0)).sum())
    fp = int(((true == 0) & (pred == 1)).sum())
    fn = int(((true == 1) & (pred == 0)).sum())
    accuracy = (tp + tn) / max(len(true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"accuracy": round(accuracy, 4), "precision": round(precision, 4),
            "recall": round(recall, 4), "f1": round(f1, 4),
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/train")
def train():
    x_train, x_test, y_train, y_test, test_frame, threshold = load_dataset()
    network = NeuralNetwork([4, 12, 8, 1], learning_rate=0.008, seed=7)
    network.fit(x_train, y_train, epochs=260, batch_size=64)
    predictions = network.predict(x_test)
    result = metrics(y_test, predictions)
    result.update({
        "samples": {"train": len(x_train), "test": len(x_test)},
        "threshold": round(threshold, 2),
        "loss": [round(value, 6) for value in network.loss_history[::5]],
        "architecture": "4 → 12 → 8 → 1",
        "source": "Brasil.IO / COVID-19 Brasil (dados estaduais)",
        "latest_date": test_frame["date"].max().strftime("%d/%m/%Y"),
    })
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
