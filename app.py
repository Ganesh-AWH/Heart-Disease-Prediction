from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
import json
from dotenv import load_dotenv
import joblib
from flask import send_from_directory

# Import custom modules
from data_preprocessing import DataPreprocessor
from supervised_models import SupervisedModels
from unsupervised_models import UnsupervisedModels
from model_comparison import ModelComparison
from supabase_client import SupabaseDatabase

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


from flask import Flask, jsonify
from supabase_client import SupabaseDatabase

db = SupabaseDatabase()

@app.route("/api/dashboard", methods=["GET"])
def get_dashboard_stats():
    try:
        stats = db.get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===============================
# Persistent model/data handling
# ===============================
MODEL_DIR = "models"
DATA_CACHE_PATH = os.path.join(MODEL_DIR, "cached_dataset.csv")
os.makedirs(MODEL_DIR, exist_ok=True)

# Initialize components
preprocessor = DataPreprocessor()
supervised_models = SupervisedModels()
unsupervised_models = UnsupervisedModels()
model_comparison = ModelComparison()
db = SupabaseDatabase()

trained_data = {
    'supervised': False,
    'unsupervised': False,
    'data_loaded': False
}


def load_persisted_models():
    """Load trained models and scaler from disk if they exist."""
    try:
        if os.path.exists(os.path.join(MODEL_DIR, "scaler.pkl")):
            preprocessor.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
            print("✅ Scaler loaded from disk.")

        for fname in os.listdir(MODEL_DIR):
            if fname.endswith(".pkl") and fname not in ["scaler.pkl"]:
                model_name = fname.replace(".pkl", "")
                supervised_models.trained_models[model_name] = joblib.load(os.path.join(MODEL_DIR, fname))
                print(f"✅ Loaded trained model: {model_name}")

        if supervised_models.trained_models:
            trained_data['supervised'] = True
            print("✅ Supervised models ready in memory.")

    except Exception as e:
        print(f"⚠️ Failed to load models: {e}")

# Load models when Flask starts
load_persisted_models()



trained_data = {
    'supervised': False,
    'unsupervised': False,
    'data_loaded': False
}

# Helper: expected feature order (keep in sync with training)
EXPECTED_FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# =========================================================
# 🩺 Health Check
# =========================================================
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Heart Disease Prediction API is running',
        'models_trained': trained_data
    }), 200


def to_serializable(obj):
    """Recursively convert NumPy datatypes to standard Python types."""
    import numpy as np

    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, list, tuple)):
        return [to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    else:
        return obj



# 📤 Data Upload (with local cache)
# =========================================================
@app.route('/api/data/upload', methods=['POST'])
def upload_data():
    try:
        data = request.json
        dataset = data.get('data')

        if not dataset:
            return jsonify({'error': 'No data provided'}), 400

        df = pd.DataFrame(dataset)
        data_list = df.to_dict('records')

        # ✅ Save to Supabase
        result = db.insert_heart_disease_data(data_list)
        trained_data['data_loaded'] = True

        # ✅ Save local cache for persistence
        MODEL_DIR = "models"
        os.makedirs(MODEL_DIR, exist_ok=True)
        DATA_CACHE_PATH = os.path.join(MODEL_DIR, "cached_dataset.csv")
        df.to_csv(DATA_CACHE_PATH, index=False)
        print(f"✅ Cached uploaded dataset at: {DATA_CACHE_PATH}")

        return jsonify({
            'message': 'Data uploaded successfully to Supabase and cached locally',
            'records_inserted': len(result),
            'sample': result[:5] if len(result) > 5 else result
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# =========================================================
# 🧹 Clear All Data 
# =========================================================
@app.route('/api/data/clear', methods=['DELETE'])
def clear_all_data():
    """
    Clears all Supabase tables: heart_disease_data, predictions, model_training_results.
    Also removes any local cached dataset.
    """
    try:
        tables = ['heart_disease_data', 'predictions', 'model_training_results']
        for table in tables:
            # ✅ Proper Supabase delete syntax — delete all rows safely
            response = db.client.table(table).delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
            print(f"🧹 Cleared {len(response.data) if response.data else 0} rows from {table}")

        # ✅ Delete local cached dataset (if it exists)
        cache_path = os.path.join("models", "cached_dataset.csv")
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print("🧹 Deleted local cached dataset.")

        return jsonify({'message': 'All Supabase tables and cached data cleared successfully'}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500



# =========================================================
# 🎯 Supervised Training
# =========================================================
@app.route('/api/supervised/train', methods=['POST'])
def train_supervised_models():
    try:
        data_from_db = db.get_all_heart_disease_data()

        if not data_from_db:
            return jsonify({'error': 'No data available. Please upload data first.'}), 400

        df = pd.DataFrame(data_from_db)
        df = df.drop(columns=['id', 'created_at'], errors='ignore')

        # Prepare data (this will fit scaler and save scaler.pkl via DataPreprocessor.scale_features)
        X_train_scaled, X_test_scaled, y_train, y_test, X_train_df, X_test_df = preprocessor.prepare_data(df)

        # Train models and save them automatically
        results = supervised_models.train_all_models(X_train_scaled, y_train, X_test_scaled, y_test)
        supervised_models.latest_results = results

        # Ensure scaler is saved (preprocessor.scale_features already saves scaler when fitting, but double-check)
        os.makedirs("models", exist_ok=True)
        try:
            joblib.dump(preprocessor.scaler, "models/scaler.pkl")
        except Exception:
            pass

        # Save training results to Supabase (if desired)
        for model_name, metrics in results.items():
            try:
                db.save_training_results(
                    model_type='supervised',
                    model_name=model_name,
                    metrics=metrics,
                    training_samples=len(data_from_db)
                )
            except Exception:
                pass

        # Save locally for persistence
        with open("supervised_results.json", "w") as f:
            json.dump(results, f, indent=4)

        comparison = supervised_models.compare_models()
        best_model, best_score = supervised_models.get_best_model('accuracy')
        trained_data['supervised'] = True

        return jsonify({
            'message': 'Supervised models trained successfully',
            'results': results,
            'comparison': comparison,
            'best_model': {
                'name': best_model,
                'accuracy': best_score
            }
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500




# =========================================================
# 🧮 Supervised Prediction (robust, cleaned)
# =========================================================
@app.route('/api/supervised/predict', methods=['POST'])
def predict_supervised():
    try:
        # Accept multiple input keys for robustness
        data = request.get_json(force=True)

        model_name = data.get('model_name') or data.get('model') or 'random_forest'
        # Accept either 'sample' (dict or list), 'input_data' (list or list-of-lists), or 'sample_data'
        sample_payload = data.get('sample') or data.get('input_data') or data.get('sample_data') or data.get('sample_list')

        if sample_payload is None:
            return jsonify({'error': 'No sample data provided. Send "sample" or "input_data" in JSON body.'}), 400

        # Normalize payload into a single-row DataFrame:
        # handle: list-of-lists, flat list, dict, list-of-dicts
        sample_df = None

        if isinstance(sample_payload, list):
            if len(sample_payload) == 0:
                return jsonify({'error': 'Empty sample list provided'}), 400

            first = sample_payload[0]
            if isinstance(first, list):
                # list-of-lists -> take first row
                sample_list = first
                sample_df = pd.DataFrame([sample_list], columns=EXPECTED_FEATURES[:len(sample_list)])
            elif isinstance(first, dict):
                # list-of-dicts -> take first dict
                sample_df = pd.DataFrame([first])
            else:
                # flat list provided as top-level list
                # e.g. input_data: [a,b,c,...]
                sample_list = sample_payload
                sample_df = pd.DataFrame([sample_list], columns=EXPECTED_FEATURES[:len(sample_list)])
        elif isinstance(sample_payload, dict):
            sample_df = pd.DataFrame([sample_payload])
        else:
            # single flat list (not wrapped)
            sample_list = sample_payload
            sample_df = pd.DataFrame([sample_list], columns=EXPECTED_FEATURES[:len(sample_list)])

        # Ensure consistent feature order and fill missing features with 0
        sample_df = sample_df.reindex(columns=EXPECTED_FEATURES, fill_value=0)

        # Load scaler (must exist from training)
        scaler_path = "models/scaler.pkl"
        if not os.path.exists(scaler_path):
            return jsonify({'error': 'Scaler not found. Please train models first via /api/supervised/train.'}), 400
        scaler = joblib.load(scaler_path)

        # Convert to numpy values to avoid sklearn warning about feature names
        X_vals = sample_df.values.astype(float)
        X_scaled = scaler.transform(X_vals)

        # Load model from disk if not in memory
        if model_name not in supervised_models.trained_models:
            model_path = os.path.join("models", f"{model_name}.pkl")
            if not os.path.exists(model_path):
                return jsonify({'error': f'Model file not found: {model_path}. Retrain models.'}), 400
            mdl = joblib.load(model_path)
            supervised_models.trained_models[model_name] = mdl

        # Do prediction
        result = supervised_models.predict_single(model_name, X_scaled[0])

        # --- Enhanced 3-level risk logic (probability-aware) ---
        probability = result.get('probability')
        if probability:
            model_obj = supervised_models.trained_models[model_name]
            classes = getattr(model_obj, "classes_", [0, 1])

            # Find which index represents disease = 1
            disease_index = np.where(classes == 1)[0][0] if 1 in classes else 1

            prob_positive = float(probability[1])  # probability for class 1 (disease)
            result['prediction'] = int(np.argmax(probability))  # respect model's output

            if prob_positive < 0.65:
                result['risk_level'] = 'Low Risk'
            elif 0.65 <= prob_positive < 0.8:
                result['risk_level'] = 'Moderate Risk'
            else:
                result['risk_level'] = 'High Risk'
        else:
            result['risk_level'] = 'Unknown'


        # Normalize probability output keyed by class labels if available
        prob_out = None
        if probability:
            model_obj = supervised_models.trained_models[model_name]
            classes = getattr(model_obj, "classes_", None)
            if classes is not None:
                prob_out = {str(classes[i]): float(probability[i]) for i in range(len(probability))}
            else:
                prob_out = probability

        # Save prediction to DB (best-effort)
        try:
            db.save_prediction(
                model_name=model_name,
                input_data=sample_df.iloc[0].to_dict(),
                prediction=int(result.get('prediction', 0)),
                probability=prob_out
            )
        except Exception:
            pass

        return jsonify({
            'model': model_name,
            'prediction': result,
            'probability': prob_out
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# =========================================================
# 📊 Feature Importance for Supervised Models
# =========================================================
@app.route('/api/supervised/feature-importance', methods=['GET'])
def get_feature_importance():
    try:
        model_name = request.args.get('model', 'random_forest')

        # Ensure model is loaded
        if model_name not in supervised_models.trained_models:
            model_path = os.path.join("models", f"{model_name}.pkl")
            if not os.path.exists(model_path):
                return jsonify({'error': f'Model file not found: {model_path}. Retrain models first.'}), 400
            supervised_models.trained_models[model_name] = joblib.load(model_path)

        model = supervised_models.trained_models[model_name]

        # Load cached dataset for correlation-based importance
        if os.path.exists("models/cached_dataset.csv"):
            df = pd.read_csv("models/cached_dataset.csv")
        else:
            data_from_db = db.get_all_heart_disease_data()
            if not data_from_db:
                return jsonify({"error": "No data available for computing feature importance"}), 400
            df = pd.DataFrame(data_from_db)

        # Drop non-feature columns
        df = df.drop(columns=["id", "created_at"], errors="ignore")

        # Expected features (same as training)
        features = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]

        # Ensure all expected features exist in data
        df = df.reindex(columns=features + (['target'] if 'target' in df.columns else []), fill_value=0)

        # =========================
        # 1️⃣ Model-based importance
        # =========================
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])

        # =========================
        # 2️⃣ Fallback: Correlation-based importance
        # =========================
        else:
            if "target" in df.columns:
                corr = df[features + ["target"]].corr()["target"].abs()
                corr = corr.drop("target", errors="ignore")

                # Normalize correlation importances to 0–1
                if corr.sum() > 0:
                    importances = corr.values / corr.sum()
                else:
                    importances = np.zeros(len(features))
            else:
                importances = np.zeros(len(features))

        # Sort + format output
        importance_data = sorted(
            [{"feature": f, "importance": float(i)} for f, i in zip(features, importances)],
            key=lambda x: x["importance"],
            reverse=True,
        )

        return jsonify({
            "model": model_name,
            "feature_importance": importance_data
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500







# =========================================================
# 🔍 Unsupervised Training
# =========================================================
@app.route('/api/unsupervised/train', methods=['POST'])
def train_unsupervised_models():
    try:
        data = request.json
        n_clusters = data.get('n_clusters', 2)
        data_from_db = db.get_all_heart_disease_data()

        if not data_from_db:
            return jsonify({'error': 'No data available. Please upload data first.'}), 400

        df = pd.DataFrame(data_from_db)
        df_unsup = df.drop(columns=['target'], errors='ignore')
        preprocessor.feature_names = df_unsup.columns.tolist()
        X_scaled = preprocessor.prepare_unsupervised_data(df_unsup)

        # Train all models
        results = unsupervised_models.train_all_models(X_scaled, n_clusters=n_clusters)
        unsupervised_models.latest_results = results

        # Save results to Supabase
        for model_name, metrics in results.items():
            db.save_training_results(
                model_type='unsupervised',
                model_name=model_name,
                metrics=to_serializable(metrics),  # ✅ ensure JSON-safe metrics
                training_samples=len(data_from_db)
            )

        # Save local JSON file
        with open("unsupervised_results.json", "w") as f:
            json.dump(to_serializable(results), f, indent=4)

        # Comparison + Best Model
        comparison = unsupervised_models.compare_models()
        best_model, best_score = unsupervised_models.get_best_model('silhouette_score')

        # Cluster distributions
        cluster_distributions = {}
        for model_name in results.keys():
            cluster_distributions[model_name] = unsupervised_models.get_cluster_distribution(model_name)

        trained_data['unsupervised'] = True

        # ✅ Return fully serializable JSON response
        return jsonify(to_serializable({
            'message': 'Unsupervised models trained successfully',
            'results': results,
            'comparison': comparison,
            'best_model': {
                'name': best_model,
                'silhouette_score': best_score
            },
            'cluster_distributions': cluster_distributions
        })), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/unsupervised/predict', methods=['POST'])
def predict_unsupervised():
    try:
        # ✅ Initialize preprocessor here
        preprocessor = DataPreprocessor()

        if not trained_data['unsupervised']:
            data_from_db = db.get_all_heart_disease_data()
            if not data_from_db:
                return jsonify({'error': 'No data available. Upload data first.'}), 400

            df = pd.DataFrame(data_from_db)
            X_scaled = preprocessor.prepare_unsupervised_data(df)
            unsupervised_models.train_all_models(X_scaled, n_clusters=2)
            trained_data['unsupervised'] = True

        data = request.json
        model_name = data.get('model_name', 'kmeans')
        sample_data = data.get('sample')

        if not sample_data:
            return jsonify({'error': 'No sample data provided'}), 400

        X_scaled = preprocessor.preprocess_single_sample(sample_data)
        cluster = unsupervised_models.predict_cluster(model_name, X_scaled)

        db.save_prediction(
            model_name=model_name,
            input_data=sample_data,
            cluster=int(cluster[0])
        )

        return jsonify({
            'model': model_name,
            'cluster': int(cluster[0]),
            'cluster_label': f"Cluster {cluster[0]}"
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# =========================================================
# 📊 Comparison Report
# =========================================================
@app.route('/api/comparison/report', methods=['GET'])
def get_model_comparison():
    try:
        sup_results = getattr(supervised_models, 'latest_results', None)
        unsup_results = getattr(unsupervised_models, 'latest_results', None)

        # Try DB fallback
        if not sup_results:
            sup_data = db.get_training_results(model_type='supervised')
            sup_results = {
                item['model_name']: item.get('metadata', {}) for item in sup_data
            } if sup_data else None

        if not unsup_results:
            unsup_data = db.get_training_results(model_type='unsupervised')
            unsup_results = {
                item['model_name']: item.get('metadata', {}) for item in unsup_data
            } if unsup_data else None

        # ✅ Final fallback: read from JSON file
        if not sup_results and not unsup_results:
            import json, os
            report_path = os.path.join(os.getcwd(), "model_comparison_report.json")
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    report = json.load(f)
                return jsonify(report), 200
            else:
                return jsonify({'error': 'No comparison data found (train models first).'}), 400

        # Generate combined report from live data
        report = model_comparison.generate_comprehensive_report(
            supervised_results=sup_results,
            unsupervised_results=unsup_results
        )

        return jsonify(report), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/statistics', methods=['GET'])
def get_data_statistics():
    try:
        data_from_db = db.get_all_heart_disease_data()

        if not data_from_db:
            return jsonify({'error': 'No data available in database.'}), 400

        df = pd.DataFrame(data_from_db)

        # Drop technical columns
        df = df.drop(columns=['id', 'created_at'], errors='ignore')

        # Compute statistics
        stats = {
            'total_records': len(df),
            'average_age': round(df['age'].mean(), 2) if 'age' in df.columns else None,
            'average_chol': round(df['chol'].mean(), 2) if 'chol' in df.columns else None,
            'average_trestbps': round(df['trestbps'].mean(), 2) if 'trestbps' in df.columns else None,
        }

        # Target distribution (if available)
        if 'target' in df.columns:
            counts = df['target'].value_counts().to_dict()
            stats['target_distribution'] = {str(k): int(v) for k, v in counts.items()}

        return jsonify(stats), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500




# =========================================================
# 🚀 Run Flask App
# =========================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5003))
    app.run(host='0.0.0.0', port=port, debug=True)
