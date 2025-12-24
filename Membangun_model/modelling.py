import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load preprocessed data
data_path = os.path.join(os.path.dirname(__file__), '..', 'preprocessing', 'processed_data.csv')
if not os.path.exists(data_path):
    data_path = '../preprocessed_data.csv'

df = pd.read_csv(data_path)
print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Prepare features and target
X = df.drop('Target', axis=1)
y = df['Target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set MLflow experiment
mlflow.set_experiment("Basic_Model_Training")

# Enable autologging
mlflow.sklearn.autolog()

# Start MLflow run
with mlflow.start_run(run_name="RandomForest_Basic_Autolog"):
    # Train model
    print("Training model with autolog...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics (autolog will log these automatically)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("=" * 50)
    print("BASIC MODEL TRAINING RESULTS (Autolog)")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("=" * 50)
    print("✓ Training completed successfully!")
    print("✓ Check MLflow UI at http://127.0.0.1:5000")