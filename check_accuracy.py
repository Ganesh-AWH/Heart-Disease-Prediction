# check_accuracy.py
import pandas as pd
from data_preprocessing import DataPreprocessor
from supervised_models import SupervisedModels
import joblib
import os

print("\n=== Checking Model Accuracies ===\n")

preprocessor = DataPreprocessor()
df = pd.read_csv("heart.csv")
X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)

models = SupervisedModels()
model_dir = "models_optimized"

# Load optimized models if available
for model_name in models.models.keys():
    model_path = os.path.join(model_dir, f"{model_name}_optimized.pkl")
    if os.path.exists(model_path):
        models.trained_models[model_name] = joblib.load(model_path)
        print(f"✅ Loaded optimized model: {model_name}")
    else:
        print(f"⚠️ Model not found, skipping: {model_name}")

# Evaluate all models
print("\n--- Model Performance ---")
for model_name, model in models.trained_models.items():
    metrics = models.evaluate_model(model_name, X_test, y_test)
    print(f"\nModel: {model_name}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-Score : {metrics['f1_score']:.4f}")
    print("-" * 50)
