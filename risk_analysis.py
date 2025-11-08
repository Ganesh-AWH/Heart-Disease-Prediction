import pandas as pd
import os
import joblib
from pprint import pprint

def get_risk_ranges(model, X):
    # Try using predict_proba for probability if available
    try:
        y_pred_prob = model.predict_proba(X)[:, 1]  # probability of high risk
        y_pred = (y_pred_prob >= 0.5).astype(int)
    except:
        y_pred = model.predict(X)
    
    df_temp = X.copy()
    df_temp['risk'] = y_pred
    df_temp['risk_label'] = df_temp['risk'].map({0: 'Low Risk', 1: 'High Risk'})
    
    # Feature ranges per risk
    risk_summary = {}
    for risk in ['Low Risk', 'High Risk']:
        risk_data = df_temp[df_temp['risk_label'] == risk]
        risk_summary[risk] = {col: (risk_data[col].min(), risk_data[col].max()) for col in X.columns}
    
    return risk_summary

def main():
    # 1️⃣ Load dataset
    df = pd.read_csv("heart.csv")
    X = df.drop(columns=['target'])  # features
    y = df['target']                 # target

    # 2️⃣ Load all supervised models from models folder
    supervised_model_files = [
        'random_forest.pkl',
        'decision_tree.pkl',
        'logistic_regression.pkl',
        'svm.pkl'
    ]

    supervised_models = {}
    for file in supervised_model_files:
        model_name = file.replace(".pkl", "")
        path = os.path.join("models", file)
        if os.path.exists(path):
            supervised_models[model_name] = joblib.load(path)
        else:
            print(f"⚠️ Model file {file} not found!")

    # 3️⃣ Get feature ranges per risk
    for model_name, model in supervised_models.items():
        summary = get_risk_ranges(model, X)
        print(f"\n=== {model_name} Feature Ranges by Risk ===")
        pprint(summary)

# Run main method
if __name__ == "__main__":
    main()
