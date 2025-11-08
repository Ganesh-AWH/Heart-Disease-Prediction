from supervised_models import SupervisedModels
import joblib
import numpy as np
import os

models = SupervisedModels()
model_dir = "models_optimized"

# Load all optimized models
for model_name in models.models.keys():
    path = os.path.join(model_dir, f"{model_name}_optimized.pkl")
    if os.path.exists(path):
        models.trained_models[model_name] = joblib.load(path)
        print(f"✅ Loaded: {model_name}")

# Test patient data
samples = {
    "Healthy_Person": [45, 0, 0, 120, 200, 0, 1, 170, 0, 0.5, 1, 0, 1],
    "Moderate_Risk_Person": [55, 1, 2, 140, 240, 0, 1, 160, 0, 0.8, 1, 0, 2],
    "High_Risk_Person": [70, 1, 3, 160, 300, 1, 1, 150, 0, 1.0, 1, 1, 2],
}

print("\n=== Testing Model Predictions ===")
for label, values in samples.items():
    print(f"\n👤 {label} Input: {values}")
    for model_name in ["decision_tree", "random_forest", "svm", "logistic_regression", "knn"]:
        result = models.predict_single(model_name, values)
        print(f"{model_name:20} → {result['risk_level']:<15} | "
              f"Prediction: {'Disease' if result['prediction']==1 else 'No Disease'} "
              f"({result['probability'][1]*100:.2f}% Disease probability)")
    print("-" * 90)
