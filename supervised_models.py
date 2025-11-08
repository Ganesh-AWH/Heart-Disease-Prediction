import numpy as np
import os
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import json


class SupervisedModels:
    def __init__(self):
        # Define all models
        self.models = {
            'decision_tree': DecisionTreeClassifier(
                max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5, random_state=42
            ),
            'svm': SVC(
                kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42
            ),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
        }

        self.trained_models = {}
        self.model_results = {}
        self.latest_results = None

        # Ensure model folder exists
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)

    # 🧠 Train a single model (with optional SMOTE)
    def train_model(self, model_name, X_train, y_train, use_smote=False):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        y_train = np.array(y_train).astype(int).ravel()

        # Optional: balance data with SMOTE
        if use_smote:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"✅ Used SMOTE. Resampled data: {X_train.shape}, class balance: {np.bincount(y_train)}")
        else:
            print("✅ Training on real dataset (no SMOTE).")

        model = self.models[model_name]
        model.fit(X_train, y_train)

        # Save trained model
        self.trained_models[model_name] = model
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        print(f"✅ Trained and saved model: {model_path}")

        return model

    # 🔁 Train all models
    def train_all_models(self, X_train, y_train, X_test, y_test, use_smote=False):
        results = {}
        for model_name in self.models.keys():
            print(f"🚀 Training {model_name}...")
            self.train_model(model_name, X_train, y_train, use_smote=use_smote)

            y_test_arr = np.array(y_test).astype(int)
            y_pred = self.trained_models[model_name].predict(X_test)

            metrics = {
                'accuracy': float(accuracy_score(y_test_arr, y_pred)),
                'precision': float(precision_score(y_test_arr, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test_arr, y_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_test_arr, y_pred, average='weighted', zero_division=0)),
                'confusion_matrix': confusion_matrix(y_test_arr, y_pred).tolist(),
                'classification_report': classification_report(y_test_arr, y_pred, output_dict=True, zero_division=0)
            }
            self.model_results[model_name] = metrics
            results[model_name] = metrics

        self.latest_results = results
        return results

    # 🔍 Evaluate single model
    def evaluate_model(self, model_name, X_test, y_test):
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")

        y_test_arr = np.array(y_test).astype(int)
        y_pred = self.trained_models[model_name].predict(X_test)

        metrics = {
            'accuracy': float(accuracy_score(y_test_arr, y_pred)),
            'precision': float(precision_score(y_test_arr, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test_arr, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test_arr, y_pred, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test_arr, y_pred).tolist(),
            'classification_report': classification_report(y_test_arr, y_pred, output_dict=True, zero_division=0)
        }
        self.model_results[model_name] = metrics
        return metrics

    # 📊 Compare model performances
    def compare_models(self):
        return {
            m: {k: v[k] for k in ['accuracy', 'precision', 'recall', 'f1_score']}
            for m, v in self.model_results.items()
        }

    def get_best_model(self, metric='accuracy'):
        if not self.model_results:
            return None, None
        best_model = max(self.model_results.items(), key=lambda x: x[1][metric])
        return best_model[0], best_model[1][metric]

    # 🔮 Predict single sample
    def predict_single(self, model_name, sample):
        # Load model if not already in memory
        if model_name not in self.trained_models:
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            if os.path.exists(model_path):
                self.trained_models[model_name] = joblib.load(model_path)
            else:
                raise ValueError(f"Model {model_name} not trained or file missing")

        model = self.trained_models[model_name]
        sample_arr = np.array(sample).reshape(1, -1)

        prediction = int(model.predict(sample_arr)[0])
        probability = (
            model.predict_proba(sample_arr)[0]
            if hasattr(model, 'predict_proba')
            else [0.5, 0.5]
        )

        prob_1 = probability[1]
        if prob_1 >= 0.75:
            risk_level = "High Risk"
        elif prob_1 >= 0.45:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"

        return {
            'prediction': prediction,
            'probability': probability.tolist(),
            'risk_level': risk_level
        }

    # 💾 Save/load
    def save_model(self, model_name, filepath):
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.trained_models[model_name], filepath)

    def load_model(self, model_name, filepath):
        model = joblib.load(filepath)
        self.trained_models[model_name] = model
        return model
    
    def load_trained_models(self):
        """Load all trained models saved as .pkl files from the models/ directory"""
        import os
        import joblib

        model_files = [f for f in os.listdir(self.model_dir) if f.endswith(".pkl")]
        if not model_files:
            print("⚠️ No trained models found in 'models/' folder.")
            return

        for file in model_files:
            model_name = file.replace(".pkl", "")
            model_path = os.path.join(self.model_dir, file)
            try:
                self.trained_models[model_name] = joblib.load(model_path)
                print(f"✅ Loaded model: {model_name}")
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")



if __name__ == "__main__":
    from data_preprocessing import DataPreprocessor
    import pandas as pd

    print("\n=== Running Supervised Models ===\n")
    preprocessor = DataPreprocessor()

    df = pd.read_csv("heart.csv")
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)

    models = SupervisedModels()
    results = models.train_all_models(X_train, y_train, X_test, y_test, use_smote=False)

     # 💾 Save results
    with open("supervised_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("✅ Saved results to supervised_results.json")


    print("\n--- Model Evaluation Results ---\n")
    for model_name, metrics in results.items():
        print(f"Model: {model_name}")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1-Score : {metrics['f1_score']:.4f}")
        print("-" * 40)

