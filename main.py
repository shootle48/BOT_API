import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

final_model = joblib.load("best_diabetes_model.pkl")
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
        data = request.get_json()
        print(f"🔹 Received Data: {data}")

        # รับข้อมูลและตรวจสอบค่าที่ต้องการ
        Glucose = float(data["Glucose"])
        Insulin = float(data["Insulin"])
        BMI = float(data["BMI"])
        print(f"Features: Glucose={Glucose}, Insulin={Insulin}, BMI={BMI}")

        # เตรียมข้อมูลสำหรับทำนาย
        input_features = np.array([[Glucose, Insulin, BMI]])
        prediction = model.predict(input_features)
        outcome = "เป็นเบาหวาน" if prediction[0] == 1 else "ไม่เป็นเบาหวาน"
        prediction_class = "positive" if prediction[0] == 1 else "negative"

        print(f"Prediction: {outcome}")
        return jsonify({
            "prediction": outcome,
            "prediction_class": prediction_class
        })
    
    except Exception as e:
        print(f"Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
