# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
df = pd.read_csv("kidney_disease.csv")

# Print first few rows to check structure
print("Dataset Preview:")
print(df.head())

# Strip spaces from column names to avoid mismatches
df.columns = df.columns.str.strip()

# Rename 'classification' column to 'class' for consistency
if 'classification' in df.columns:
    df.rename(columns={'classification': 'class'}, inplace=True)

# Check column names
print("\nAvailable columns in dataset:")
print(df.columns)

# Handle missing or incorrect column names dynamically
expected_num_cols = ['age', 'bp', 'sg', 'al', 'su']
num_cols = [col for col in expected_num_cols if col in df.columns]  # Only include existing columns

# Convert numeric columns to float
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values dynamically
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

# Apply numerical imputer
num_cols_in_df = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols_in_df] = num_imputer.fit_transform(df[num_cols_in_df])

# Apply categorical imputer
cat_cols_in_df = df.select_dtypes(include=['object']).columns
df[cat_cols_in_df] = df[cat_cols_in_df].apply(lambda x: cat_imputer.fit_transform(x.values.reshape(-1,1)).flatten())

# Encode categorical variables
encoder = LabelEncoder()
for col in cat_cols_in_df:
    df[col] = encoder.fit_transform(df[col])

# Ensure 'class' column exists
if 'class' not in df.columns:
    raise ValueError("Error: 'class' column is missing from dataset!")

# Split features and target
X = df.drop(columns=['class'])  # Features
y = df['class']  # Target

# Split dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🔹 Model Accuracy: {accuracy:.2f}")

print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues", fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Save the trained model
joblib.dump(model, "ckd_model.pkl")
print("\n✅ Model saved as 'ckd_model.pkl'.")
