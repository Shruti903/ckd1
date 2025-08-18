import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load("ckd_model.pkl")
scaler = joblib.load("scaler.pkl")

# Function to make predictions
def predict_ckd():
    try:
        # Get values from input fields
        age =float(entry_age.get())
        bp = float(entry_bp.get())
        sg = float(entry_sg.get())
        al = int(entry_al.get())
        su = int(entry_su.get())

        # Prepare input data
        new_patient = np.array([[age, bp, sg, al, su]])
        new_patient_scaled = scaler.transform(new_patient)

        # Predict
        prediction = model.predict(new_patient_scaled)
        proba = model.predict_proba(new_patient_scaled)  

        # Show result with probability
        if prediction[0] == 1:
            messagebox.showwarning("Prediction", f"🔴 CKD Detected! (Risk: {proba[0][1]*100:.2f}%)")
        else:
            messagebox.showinfo("Prediction", f"🟢 No CKD Detected (Confidence: {proba[0][0]*100:.2f}%)")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values!")

# Create GUI Window
root = tk.Tk()
root.title("CKD Prediction System")
root.geometry("400x450")

# Labels and Entry Fields
tk.Label(root, text="Enter Patient Details", font=("Arial", 14, "bold")).pack(pady=10)
tk.Label(root, text="Age").pack()
entry_age = tk.Entry(root)
entry_age.pack()

tk.Label(root, text="Blood Pressure").pack()
entry_bp = tk.Entry(root)
entry_bp.pack()

tk.Label(root, text="Specific Gravity").pack()
entry_sg = tk.Entry(root)
entry_sg.pack()

tk.Label(root, text="Albumin Level").pack()
entry_al = tk.Entry(root)
entry_al.pack()

tk.Label(root, text="Sugar Level").pack()
entry_su = tk.Entry(root)
entry_su.pack()

# Predict Button
tk.Button(root, text="Predict CKD", command=predict_ckd, bg="blue", fg="white", font=("Arial", 12, "bold")).pack(pady=20)

# Run GUI
root.mainloop()


