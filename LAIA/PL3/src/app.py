"""
Deployment scaffold for the predictive maintenance model — same shape as
app.py from Tutorial 2 (Flask, model.pkl loaded with joblib, JSON in,
JSON out), extended with a batch endpoint like the one you built in
Tutorial 2, Exercise B.

Unlike Tutorial 2's run.sh, this app does NOT train a model on startup —
you already did that in the notebook (Parts 1-6). Save your best model as
model.pkl before you build the image:

    import joblib
    joblib.dump(best_model, "model.pkl")

Then, exactly like Tutorial 2:

    docker build -t predictive-maintenance-app .
    docker run -p 8000:8000 predictive-maintenance-app

Single-reading request:
    curl -X POST -H "Content-Type: application/json" \\
      -d '{"Type": "M", "Air temperature [K]": 300.5, "Process temperature [K]": 310.2,
           "Rotational speed [rpm]": 1450, "Torque [Nm]": 42.1, "Tool wear [min]": 120}' \\
      http://localhost:8000/predict

Batch request (this is the one you'll use on future_batch.csv):
    curl -X POST -H "Content-Type: application/json" \\
      -d '{"readings": [{"Type": "M", "Air temperature [K]": 300.5, ...}, {...}]}' \\
      http://localhost:8000/predict-batch
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd

FEATURES_NUM = [
    "Air temperature K",
    "Process temperature K",
    "Rotational speed rpm",
    "Torque Nm",
    "Tool wear min",
]

# Load trained model
model = joblib.load("../results/model.pkl")

app = Flask(__name__)


def readings_to_features(readings):
    df = pd.DataFrame(readings)
    
    # Limpar os nomes das colunas pro XGBoost não reclamar
    df.columns = df.columns.str.replace(r'[\[\]<]', '', regex=True)
    
    # Criar apenas as variáveis dummy que o modelo realmente usou no treino
    df["Type_L"] = (df["Type"] == "L").astype(int)
    df["Type_M"] = (df["Type"] == "M").astype(int)
    
    # Retornar o DataFrame SEM o "Type_H"
    return df[FEATURES_NUM + ["Type_L", "Type_M"]]


@app.route("/predict", methods=["POST"])
def predict():
    reading = request.json
    X = readings_to_features([reading])
    proba = float(model.predict_proba(X)[0, 1])
    return jsonify({
        "failure_probability": round(proba, 4),
        "prediction": int(proba >= 0.5),
    })


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    readings = request.json["readings"]
    X = readings_to_features(readings)
    probas = model.predict_proba(X)[:, 1]
    return jsonify({
        "failure_probabilities": [round(float(p), 4) for p in probas],
        "predictions": [int(p >= 0.5) for p in probas],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
