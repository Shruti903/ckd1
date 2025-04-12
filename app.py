from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("ckd_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input values
        age = float(request.form["age"])
        bp = float(request.form["bp"])
        sg = float(request.form["sg"])
        al = int(request.form["al"])
        su = int(request.form["su"])

        # Prepare input for model
        new_patient = np.array([[age, bp, sg, al, su]])
        new_patient_scaled = scaler.transform(new_patient)

        # Make prediction
        prediction = model.predict(new_patient_scaled)
        proba = model.predict_proba(new_patient_scaled)

        # Return JSON response
        if prediction[0] == 1:
            return jsonify({"status": "🔴 CKD Detected!", "risk": f"Risk: {proba[0][1]*100:.2f}%"})
        else:
            return jsonify({"status": "🟢 No CKD Detected", "risk": f"Confidence: {proba[0][0]*100:.2f}%"})

    except ValueError:
        return jsonify({"error": "Invalid input! Please enter valid numeric values."})

if __name__ == "__main__":
    app.run(debug=True)
