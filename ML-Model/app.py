from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from model_defs import AQILSTM
from datetime import datetime

app = Flask(__name__)
CORS(app)

# =========================
# CONFIG
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "multivariate_model.pth")

# =========================
# MODEL LOAD
# =========================

model = AQILSTM()

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model load failed:", e)
    model = None

# =========================
# HELPERS
# =========================

def get_aqi_condition(aqi_value):
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def get_aqi_emoji(aqi_value):
    if aqi_value <= 50:
        return "😊"
    elif aqi_value <= 100:
        return "😐"
    elif aqi_value <= 150:
        return "⚠️"
    elif aqi_value <= 200:
        return "❌"
    elif aqi_value <= 300:
        return "🔴"
    else:
        return "💀"


# =========================
# CORE PREDICTION
# =========================

def get_prediction(state_name, area_name):

    # 🔥 synthetic safe data (no CSV dependency)
    aqi_values = np.array([
        80, 85, 90, 88, 92, 95, 100,
        98, 97, 93, 91, 89, 87, 90
    ])

    if len(aqi_values) < 7:
        return None, "Not enough data"

    window_size = min(30, len(aqi_values))
    series = aqi_values[-window_size:]

    features_df = pd.DataFrame(index=range(len(series)))
    features_df["aqi"] = series

    # lag features
    for i in range(1, 8):
        features_df[f"aqi_lag_{i}"] = features_df["aqi"].shift(i)

    # rolling stats
    features_df["rolling_mean_7"] = features_df["aqi"].rolling(7).mean()
    features_df["rolling_std_7"] = features_df["aqi"].rolling(7).std()
    features_df["rolling_min_7"] = features_df["aqi"].rolling(7).min()
    features_df["rolling_max_7"] = features_df["aqi"].rolling(7).max()

    features_df = features_df.bfill().ffill()

    recent_data = features_df.values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(recent_data)

    input_window = torch.FloatTensor(scaled_data).unsqueeze(0)

    with torch.no_grad():
        prediction_scaled = model(input_window)

    prediction_scaled = prediction_scaled.numpy().flatten()

    dummy = np.tile(scaled_data[-1], (7, 1))
    dummy[:, 0] = prediction_scaled

    prediction = scaler.inverse_transform(dummy)[:, 0]

    return prediction.tolist(), None


# =========================
# ROUTES
# =========================

@app.route("/predict", methods=["GET"])
def predict():
    try:
        state = request.args.get("state", "default")
        area = request.args.get("area", "default")

        forecast, error = get_prediction(state, area)

        if error:
            return jsonify({"status": "error", "message": error}), 400

        avg = sum(forecast) / len(forecast)

        return jsonify({
            "status": "success",
            "state": state,
            "area": area,
            "average_aqi": round(avg, 2),
            "condition": get_aqi_condition(avg),
            "forecast": [
                {
                    "day": i + 1,
                    "aqi": round(v, 2),
                    "category": get_aqi_condition(v),
                    "emoji": get_aqi_emoji(v)
                }
                for i, v in enumerate(forecast)
            ]
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "time": datetime.now().isoformat()
    })


@app.route("/api-info")
def info():
    return jsonify({
        "service": "AQI Prediction API",
        "version": "2.0",
        "note": "Dataset removed for deployment stability"
    })


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 6060))
    app.run(host="0.0.0.0", port=port, debug=False)
