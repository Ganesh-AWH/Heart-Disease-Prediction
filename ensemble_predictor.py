import joblib
import numpy as np

# Load all saved models
models = {
    "decision_tree": joblib.load("models/decision_tree.pkl"),
    "random_forest": joblib.load("models/random_forest.pkl"),
    "svm": joblib.load("models/svm.pkl"),
    "logistic_regression": joblib.load("models/logistic_regression.pkl"),
    "knn": joblib.load("models/knn.pkl"),
}

def ensemble_predict(input_data):
    """
    input_data: list or numpy array of 13 features
    returns: final prediction and confidence
    """

    predictions = []
    probabilities = []

    for name, model in models.items():
        try:
            prob = model.predict_proba([input_data])[0][1]
            pred = int(prob >= 0.5)
            predictions.append(pred)
            probabilities.append(prob)
        except Exception:
            pred = model.predict([input_data])[0]
            predictions.append(pred)
            probabilities.append(pred)

    # Majority voting
    majority_vote = int(sum(predictions) >= len(predictions) / 2)

    # Average probability confidence
    avg_confidence = np.mean(probabilities)

    if majority_vote == 1:
        risk = "Moderate to High Risk"
        final_label = "Disease"
    else:
        risk = "Low Risk"
        final_label = "No Disease"

    print("\n=== Ensemble Voting Result ===")
    print(f"Models Agree on: {sum(predictions)}/{len(predictions)} for Disease")
    print(f"Final Prediction: {final_label}")
    print(f"Risk Level: {risk}")
    print(f"Confidence: {avg_confidence * 100:.2f}%\n")

    return final_label, avg_confidence

if __name__ == "__main__":
    # Example input (you can replace with user inputs)
    test_input = [55, 1, 2, 140, 240, 0, 1, 160, 0, 0.8, 1, 0, 2]
    ensemble_predict(test_input)
