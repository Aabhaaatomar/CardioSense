import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load dataset
data = pd.read_csv("cardio_train.csv", sep=";")

# Step 2: Drop useless columns (like id)
if "id" in data.columns:
    data = data.drop("id", axis=1)

# Step 3: Define X and y
X = data.drop("cardio", axis=1)
y = data["cardio"]

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))
print("✅ Feature Importances:", model.feature_importances_)

# Step 7: Save trained model
joblib.dump(model, "model.pkl")
print("🎉 Model retrained and saved as model.pkl")
