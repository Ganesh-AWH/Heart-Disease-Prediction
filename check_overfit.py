from supervised_models import SupervisedModels
from data_preprocessing import DataPreprocessor
import pandas as pd
from sklearn.metrics import accuracy_score

print("\n--- Train vs Test Accuracy (Overfitting Check) ---\n")

# Initialize models and preprocessor
models = SupervisedModels()
models.load_trained_models()

preprocessor = DataPreprocessor()
df = pd.read_csv("heart.csv")
X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)

# Define list of supervised models (ignore unsupervised or preprocessing files)
supervised_list = ["decision_tree", "random_forest", "svm", "logistic_regression", "knn"]

for model_name, model in models.trained_models.items():
    if model_name not in supervised_list:
        print(f"⚠️ Skipping {model_name} (not a supervised classifier)")
        continue

    try:
        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Accuracy scores
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        diff = abs(train_acc - test_acc)

        # Interpret result
        if diff > 0.1:
            status = "🔴 Overfitting"
        elif diff < 0.02:
            status = "🟢 Balanced"
        else:
            status = "🟡 Slightly Overfitting"

        print(f"{model_name}: Train = {train_acc:.4f}, Test = {test_acc:.4f} → {status}")

    except ValueError as e:
        print(f"❌ {model_name}: {e}")
    except Exception as e:
        print(f"❌ Unexpected error for {model_name}: {e}")
