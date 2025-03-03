import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

final_model = joblib.load("best_model.pkl")
model = final_model["model"]
label_encoders = final_model["label_encoders"]

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "Flask API is running on Render"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # รับข้อมูลจาก request body ในรูปแบบ JSON
        data = request.get_json()
        Glucose = float(data["Glucose"])
        Insulin = float(data["Insulin"])
        BMI = float(data["BMI"])
    except (ValueError, KeyError, TypeError):
        return jsonify({"error": "Invalid input data. Please provide Glucose, Insulin, and BMI values."}), 400

    # เตรียมข้อมูลสำหรับทำนายให้เป็น array 2 มิติ
    input_features = np.array([[Glucose, Insulin, BMI]])

    # ทำนายผลด้วยโมเดลที่บันทึกไว้
    prediction = model.predict(input_features)
    outcome = "เป็นเบาหวาน" if prediction[0] == 1 else "ไม่เป็นเบาหวาน"

    return jsonify({
        "prediction": outcome,
        "prediction_class": "positive" if prediction[0] == 1 else "negative"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)