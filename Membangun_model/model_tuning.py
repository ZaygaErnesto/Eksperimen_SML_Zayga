import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SKILLED MODEL TRAINING - HYPERPARAMETER TUNING WITH MANUAL LOGGING")
print("="*70)

# Load data
print("\n1. Loading preprocessed data...")
df = pd.read_csv('../preprocessed_data.csv')

print(f"   ✓ Data loaded: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")

# Separate features and target
X = df.drop(['Target', 'Failure Type'], axis=1)
y = df['Target']

print(f"\n2. Feature and Target separation:")
print(f"   ✓ Features (X): {X.shape}")
print(f"   ✓ Target (y): {y.shape}")
print(f"   ✓ Target distribution:\n{y.value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n3. Train-Test Split:")
print(f"   ✓ X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"   ✓ X_test: {X_test.shape}, y_test: {y_test.shape}")

# Set MLflow experiment (local storage)
mlflow.set_experiment("Skilled_Model_Tuning_Local")

# IMPORTANT: Disable autologging - we will do manual logging
mlflow.sklearn.autolog(disable=True)

print("\n4. MLflow Setup:")
print(f"   ✓ Experiment: Skilled_Model_Tuning_Local")
print(f"   ✓ Tracking URI: {mlflow.get_tracking_uri()}")
print(f"   ✓ Autolog: DISABLED (Manual Logging)")

# Define hyperparameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

print(f"\n5. Hyperparameter Grid:")
for param, values in param_grid.items():
    print(f"   - {param}: {values}")

# Perform Grid Search with Cross-Validation
print(f"\n6. Starting Hyperparameter Tuning...")
print(f"   This may take a few minutes...")

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    rf, 
    param_grid, 
    cv=5, 
    scoring='accuracy', 
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

# Start MLflow run with manual logging
with mlflow.start_run(run_name="RandomForest_Tuning_Manual_Logging"):
    
    print("\n" + "="*70)
    print("TRAINING MODEL WITH GRID SEARCH CV")
    print("="*70)
    
    # Train model with hyperparameter tuning
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    print(f"\n✓ Training completed!")
    print(f"   Best CV Score: {grid_search.best_score_:.4f}")
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)
    
    # ========================================================================
    # MANUAL LOGGING - PARAMETERS (sama seperti autolog)
    # ========================================================================
    print("\n7. Logging Parameters...")
    
    # Log best hyperparameters
    mlflow.log_params(grid_search.best_params_)
    
    # Log additional parameters
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_param("n_samples_train", len(X_train))
    mlflow.log_param("n_samples_test", len(X_test))
    mlflow.log_param("scoring_metric", "accuracy")
    
    print(f"   ✓ Parameters logged")
    
    # ========================================================================
    # MANUAL LOGGING - METRICS (sama seperti autolog)
    # ========================================================================
    print("\n8. Calculating and Logging Metrics...")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Log metrics (sama seperti autolog)
    mlflow.log_metric("training_accuracy_score", grid_search.best_score_)
    mlflow.log_metric("training_score", grid_search.best_score_)  # alias
    mlflow.log_metric("test_accuracy_score", accuracy)
    mlflow.log_metric("test_precision_score", precision)
    mlflow.log_metric("test_recall_score", recall)
    mlflow.log_metric("test_f1_score", f1)
    
    # Additional metrics
    mlflow.log_metric("mean_fit_time", grid_search.cv_results_['mean_fit_time'].mean())
    mlflow.log_metric("mean_score_time", grid_search.cv_results_['mean_score_time'].mean())
    
    print(f"   ✓ Metrics logged")
    
    # ========================================================================
    # MANUAL LOGGING - MODEL (sama seperti autolog)
    # ========================================================================
    print("\n9. Logging Model...")
    
    # Log model with signature and input example
    signature = mlflow.models.infer_signature(X_train, best_model.predict(X_train))
    input_example = X_train.iloc[:5]
    
    mlflow.sklearn.log_model(
        best_model, 
        "model",
        signature=signature,
        input_example=input_example
    )
    
    print(f"   ✓ Model logged")
    
    # ========================================================================
    # MANUAL LOGGING - ARTIFACTS (sama seperti autolog)
    # ========================================================================
    print("\n10. Creating and Logging Artifacts...")
    
    # Artifact 1: Feature Importances (CSV)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    feature_importance.to_csv('feature_importance.csv', index=False)
    mlflow.log_artifact('feature_importance.csv')
    
    # Artifact 2: Confusion Matrix (Plot)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    mlflow.log_artifact('confusion_matrix.png')
    plt.close()
    
    # Artifact 3: Classification Report (TXT)
    report = classification_report(y_test, y_pred, zero_division=0)
    with open('classification_report.txt', 'w') as f:
        f.write("Classification Report\n")
        f.write("="*50 + "\n\n")
        f.write(report)
    mlflow.log_artifact('classification_report.txt')
    
    # Artifact 4: GridSearch CV Results (CSV)
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv('gridsearch_cv_results.csv', index=False)
    mlflow.log_artifact('gridsearch_cv_results.csv')
    
    # Artifact 5: Model Summary (TXT)
    with open('model_summary.txt', 'w') as f:
        f.write("Model Training Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model: RandomForestClassifier\n")
        f.write(f"Best Parameters: {grid_search.best_params_}\n\n")
        f.write(f"Training Metrics:\n")
        f.write(f"  CV Score: {grid_search.best_score_:.4f}\n\n")
        f.write(f"Test Metrics:\n")
        f.write(f"  Accuracy:  {accuracy:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall:    {recall:.4f}\n")
        f.write(f"  F1 Score:  {f1:.4f}\n\n")
        f.write(f"Data Info:\n")
        f.write(f"  Train samples: {len(X_train)}\n")
        f.write(f"  Test samples:  {len(X_test)}\n")
        f.write(f"  Features:      {X_train.shape[1]}\n")
    mlflow.log_artifact('model_summary.txt')
    
    print(f"   ✓ Artifacts logged")
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING RESULTS - SKILLED MODEL WITH MANUAL LOGGING")
    print("="*70)
    print(f"\nBest Hyperparameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  - {param}: {value}")
    
    print(f"\nCross-Validation:")
    print(f"  Best CV Score: {grid_search.best_score_:.4f}")
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    print(f"\nTop 5 Important Features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nMLflow Tracking:")
    print(f"  Experiment: Skilled_Model_Tuning_Local")
    print(f"  Run Name: RandomForest_Tuning_Manual_Logging")
    print(f"\nTo view results, run: mlflow ui")
    print(f"Then open: http://localhost:5000")
    print("="*70)

print("\n✓ Script execution completed!")
print("\nNext steps:")
print("1. Run 'mlflow ui' in terminal")
print("2. Open http://localhost:5000 in browser")
print("3. Take screenshots for submission:")
print("   - Experiment list view")
print("   - Run detail with parameters")
print("   - Run detail with metrics")
print("   - Artifacts section")