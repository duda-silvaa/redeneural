from flask import Flask, jsonify, render_template

from data_pipeline import load_dataset
from neural_network import NeuralNetwork

app = Flask(__name__)


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/api/train")
def train():
    x_train, x_test, y_train, y_test, test_frame, threshold = load_dataset()
    validation_split = int(len(x_train) * 0.8)
    network = NeuralNetwork([4, 12, 8, 1], learning_rate=0.008, seed=7)
    network.fit(
        x_train[:validation_split],
        y_train[:validation_split],
        epochs=260,
        batch_size=64,
        validation_data=(x_train[validation_split:], y_train[validation_split:]),
        early_stopping_patience=25,
        min_delta=1e-5,
    )
    predictions = network.predict(x_test)
    result = {
        key: round(value, 4) if isinstance(value, float) else value
        for key, value in network.classification_report(y_test, predictions).items()
    }
    result.update({
        "samples": {
            "train": validation_split,
            "validation": len(x_train) - validation_split,
            "test": len(x_test),
        },
        "threshold": round(threshold, 2),
        "loss": [round(value, 6) for value in network.loss_history[::5]],
        "architecture": "4 → 12 → 8 → 1",
        "parameters": network.parameter_count(),
        "epochs_trained": len(network.loss_history),
        "source": "Brasil.IO / COVID-19 Brasil (dados estaduais)",
        "latest_date": test_frame["date"].max().strftime("%d/%m/%Y"),
    })
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
