import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv("kidney_disease.csv")

# Ensure column names are lowercase and stripped of spaces
df.columns = df.columns.str.strip().str.lower()

# Check if 'classification' column exists
if 'classification' not in df.columns:
    print("❌ Error: Column 'classification' not found! Available columns:", df.columns)
    exit()

# Convert target column 'classification' to numerical values
df['classification'] = df['classification'].map({'ckd': 1, 'notckd': 0})

# Drop rows with missing values
df.dropna(inplace=True)

# Select relevant features (modify if needed)
features = ['age', 'bp', 'sg', 'al', 'su']
target = 'classification'

X = df[features]
y = df[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Standardize features
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Save the model and scaler
joblib.dump(model, "ckd_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model training completed! Model and scaler saved successfully.")
