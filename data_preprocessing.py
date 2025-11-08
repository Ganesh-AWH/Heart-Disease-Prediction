import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_names = None

    # -------------------------------------------------------------
    # Load CSV or custom data, or generate synthetic if missing
    # -------------------------------------------------------------
    def load_data(self, filepath=None, data=None):
        if filepath:
            df = pd.read_csv(filepath)
        elif data is not None:
            df = pd.DataFrame(data)
        else:
            df = self.generate_sample_data()
        return df

    # -------------------------------------------------------------
    # Generate synthetic dataset (only fallback)
    # -------------------------------------------------------------
    def generate_sample_data(self, n_samples=300):
        np.random.seed(42)
        age = np.random.randint(30, 80, n_samples)
        sex = np.random.randint(0, 2, n_samples)
        cp = np.random.randint(0, 4, n_samples)
        trestbps = np.random.randint(90, 180, n_samples)
        chol = np.random.randint(150, 320, n_samples)
        fbs = np.random.randint(0, 2, n_samples)
        restecg = np.random.randint(0, 2, n_samples)
        thalach = np.random.randint(100, 200, n_samples)
        exang = np.random.randint(0, 2, n_samples)
        oldpeak = np.random.uniform(0, 4, n_samples)
        slope = np.random.randint(0, 3, n_samples)
        ca = np.random.randint(0, 4, n_samples)
        thal = np.random.randint(0, 3, n_samples)

        risk_score = (
            0.05 * age + 0.03 * trestbps + 0.02 * chol +
            0.8 * exang + 0.5 * (cp == 3).astype(int) +
            0.5 * (oldpeak > 2).astype(int) -
            0.03 * (thalach - 150)
        )

        prob = 1 / (1 + np.exp(-0.05 * (risk_score - 16)))
        prob = np.clip(prob, 0.1, 0.9)
        target = (np.random.rand(n_samples) < prob).astype(int)

        return pd.DataFrame({
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
            'ca': ca, 'thal': thal, 'target': target
        })

    # -------------------------------------------------------------
    # Handle missing numeric values
    # -------------------------------------------------------------
    def handle_missing_values(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        return df

    # -------------------------------------------------------------
    # Encode categorical columns
    # -------------------------------------------------------------
    def encode_categorical(self, df):
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        return df

    # -------------------------------------------------------------
    # Scale features (fit + save OR load existing)
    # -------------------------------------------------------------
    def scale_features(self, X, fit=True, force_fit=False):
        os.makedirs("models", exist_ok=True)
        scaler_path = os.path.join("models", "scaler.pkl")

        X_values = X.values if hasattr(X, "values") else X
        X_values = np.array(X_values, dtype=float)

        if fit:
            # Only re-fit if scaler not found OR explicitly forced
            if not os.path.exists(scaler_path) or force_fit:
                X_scaled = self.scaler.fit_transform(X_values)
                joblib.dump(self.scaler, scaler_path)
            else:
                self.scaler = joblib.load(scaler_path)
                X_scaled = self.scaler.transform(X_values)
        else:
            if not os.path.exists(scaler_path):
                raise FileNotFoundError("❌ Scaler not found. Please train models first.")
            self.scaler = joblib.load(scaler_path)
            X_scaled = self.scaler.transform(X_values)

        return X_scaled


    # -------------------------------------------------------------
    # Prepare data for training (optional SMOTE)
    # -------------------------------------------------------------
    def prepare_data(self, df, target_column='target', test_size=0.2,
                     random_state=42, use_smote=False):
        df = df.drop(columns=['id', 'created_at'], errors='ignore')
        df = self.handle_missing_values(df)
        df = self.encode_categorical(df)

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        X = df.drop(columns=[target_column])
        y = df[target_column]
        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # ✅ Optional SMOTE (not by default)
        if use_smote:
            sm = SMOTE(random_state=random_state)
            X_train, y_train = sm.fit_resample(X_train, y_train)

        # ✅ Scale data
        X_train_scaled = self.scale_features(X_train, fit=True)
        X_test_scaled = self.scale_features(X_test, fit=False)
        
        X_train_df = pd.DataFrame(X_train, columns=X_train.columns)
        X_test_df = pd.DataFrame(X_test, columns=X_test.columns)
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values, X_train_df, X_test_df

    # -------------------------------------------------------------
    # Preprocess a single patient for prediction
    # -------------------------------------------------------------
    def preprocess_single_sample(self, sample_data):
        df = pd.DataFrame([sample_data])
        df = df.drop(columns=['id', 'created_at'], errors='ignore')
        df = self.handle_missing_values(df)
        df = self.encode_categorical(df)

        if self.feature_names is not None:
            df = df.reindex(columns=self.feature_names, fill_value=0)

        X_scaled = self.scale_features(df, fit=False)
        return X_scaled

        # -------------------------------------------------------------
    # Prepare data for unsupervised learning (no target column)
    # -------------------------------------------------------------
    def prepare_unsupervised_data(self, df):
        df = df.drop(columns=['id', 'created_at', 'target'], errors='ignore')
        df = self.handle_missing_values(df)
        df = self.encode_categorical(df)
        X_scaled = self.scale_features(df, fit=True)
        return X_scaled
