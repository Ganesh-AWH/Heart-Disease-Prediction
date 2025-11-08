# reduce_overfit.py
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from data_preprocessing import DataPreprocessor
from supervised_models import SupervisedModels
import pandas as pd

# === Load and Prepare Data ===
df = pd.read_csv("heart.csv")
preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)

# === Train Original Models ===
models = SupervisedModels()
models.train_all_models(X_train, y_train, X_test, y_test)

# === Grid Search to Reduce Overfitting ===
param_grids = {
    "decision_tree": {
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 3, 5],
    },
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 3],
    },
    "svm": {
        "C": [0.1, 0.5, 1, 2],
        "kernel": ["linear", "rbf"],
    },
    "logistic_regression": {
        "C": [0.1, 0.5, 1],
        "penalty": ["l2"],
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 9],
    },
}

print("\n=== Running GridSearchCV to Reduce Overfitting ===\n")

for name, model in models.trained_models.items():
    if name not in param_grids:
        print(f"⚠️ Skipping {name} (no param grid defined)\n")
        continue

    print(f"🔍 Tuning {name}...")
    grid = GridSearchCV(model, param_grids[name], cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    models.trained_models[name] = best_model

    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"✅ {name} optimized:")
    print(f"   Train Accuracy: {train_acc:.4f}")
    print(f"   Test Accuracy : {test_acc:.4f}")
    print(f"   Best Params   : {grid.best_params_}")
    print("-" * 50)
# === Save Optimized Models ===
import joblib
import os

save_dir = "models_optimized"
os.makedirs(save_dir, exist_ok=True)

for name, model in models.trained_models.items():
    filepath = os.path.join(save_dir, f"{name}_optimized.pkl")
    joblib.dump(model, filepath)
    print(f"💾 Saved optimized model: {filepath}")
