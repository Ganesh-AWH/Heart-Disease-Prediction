import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances_argmin_min
import skfuzzy as fuzz
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os
import json

class UnsupervisedModels:
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.model_results = {}
        self.cluster_labels = {}
        self.X_train = None  # Save training data for models like DBSCAN/Hierarchical

    # ---------------- Training Methods ---------------- #
    def train_kmeans(self, X, n_clusters=2, random_state=42):
        model = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=random_state
        )
        labels = model.fit_predict(X)
        self.trained_models['kmeans'] = model
        self.cluster_labels['kmeans'] = labels
        return model, labels

    def train_dbscan(self, X, eps=0.5, min_samples=5):
        model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        labels = model.fit_predict(X)
        self.trained_models['dbscan'] = model
        self.cluster_labels['dbscan'] = labels
        return model, labels

    def train_hierarchical(self, X, n_clusters=2, linkage='ward'):
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        labels = model.fit_predict(X)
        self.trained_models['hierarchical'] = model
        self.cluster_labels['hierarchical'] = labels
        return model, labels

    def train_fuzzy_cmeans(self, X, n_clusters=2, m=2, error=0.005, maxiter=1000):
        X_transposed = X.T
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X_transposed,
            c=n_clusters,
            m=m,
            error=error,
            maxiter=maxiter,
            init=None
        )
        labels = np.argmax(u, axis=0)
        self.trained_models['fuzzy_cmeans'] = {
            'centers': cntr,
            'membership': u,
            'n_clusters': n_clusters,
            'm': m
        }
        self.cluster_labels['fuzzy_cmeans'] = labels
        return cntr, u, labels

    def train_gmm(self, X, n_components=2, random_state=42):
        model = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            max_iter=100,
            random_state=random_state
        )
        model.fit(X)
        labels = model.predict(X)
        self.trained_models['gmm'] = model
        self.cluster_labels['gmm'] = labels
        return model, labels

    def build_autoencoder(self, input_dim, encoding_dim=8):
        input_layer = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(64, activation='relu')(input_layer)
        encoded = layers.Dense(32, activation='relu')(encoded)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        decoded = layers.Dense(32, activation='relu')(encoded)
        decoded = layers.Dense(64, activation='relu')(decoded)
        decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
        autoencoder = keras.Model(input_layer, decoded)
        encoder = keras.Model(input_layer, encoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder, encoder

    # def train_autoencoder(self, X, encoding_dim=8, epochs=50, batch_size=32):
    #     input_dim = X.shape[1]
    #     autoencoder, encoder = self.build_autoencoder(input_dim, encoding_dim)
    #     autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)
    #     encoded_data = encoder.predict(X, verbose=0)
    #     kmeans = KMeans(n_clusters=2, random_state=42)
    #     labels = kmeans.fit_predict(encoded_data)
    #     self.trained_models['autoencoder'] = {
    #         'autoencoder': autoencoder,
    #         'encoder': encoder,
    #         'kmeans': kmeans
    #     }
    #     self.cluster_labels['autoencoder'] = labels
    #     return autoencoder, encoder, labels
    
    def train_autoencoder(self, X, encoding_dim=8, epochs=50):
        import tensorflow as tf
        from sklearn.cluster import KMeans
        from keras.models import Model
        from keras.layers import Input, Dense

        input_dim = X.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')

        autoencoder.fit(X, X, epochs=epochs, batch_size=16, verbose=0)

        encoder = Model(inputs=input_layer, outputs=encoded)
        X_encoded = encoder.predict(X)

        # Run KMeans on encoded features
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(X_encoded)

        self.trained_models['autoencoder'] = kmeans
        kmeans.labels_ = kmeans.labels_  # ✅ ensures consistent access
        return autoencoder, encoder, kmeans.labels_


    # ---------------- Evaluation ---------------- #
    def evaluate_clustering(self, X, labels, model_name):
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        if n_clusters < 2 or len(labels) < 2:
            metrics = {
                'silhouette_score': 0.0,
                'davies_bouldin_score': float('inf'),
                'calinski_harabasz_score': 0.0,
                'n_clusters': n_clusters,
                'n_noise_points': np.sum(labels == -1)
            }
        else:
            valid_mask = labels >= 0
            X_valid = X[valid_mask]
            labels_valid = labels[valid_mask]
            if len(np.unique(labels_valid)) < 2:
                metrics = {
                    'silhouette_score': 0.0,
                    'davies_bouldin_score': float('inf'),
                    'calinski_harabasz_score': 0.0,
                    'n_clusters': n_clusters,
                    'n_noise_points': np.sum(labels == -1)
                }
            else:
                metrics = {
                    'silhouette_score': silhouette_score(X_valid, labels_valid),
                    'davies_bouldin_score': davies_bouldin_score(X_valid, labels_valid),
                    'calinski_harabasz_score': calinski_harabasz_score(X_valid, labels_valid),
                    'n_clusters': n_clusters,
                    'n_noise_points': np.sum(labels == -1)
                }
        self.model_results[model_name] = metrics
        return metrics
    
    def get_cluster_distribution(self, model_name):
        """
        Returns a dictionary with the count of samples in each cluster for a given model.
        Example: {'0': 45, '1': 55}
        """
        import numpy as np

        # Ensure model exists
        if model_name not in self.trained_models:
            print(f"⚠️ Model '{model_name}' not found in trained models.")
            return {}

        model = self.trained_models[model_name]

        try:
            # Handle different clustering algorithms
            if hasattr(model, 'labels_'):
                labels = model.labels_
            elif hasattr(model, 'predict'):
                labels = model.predict(self.X_train)
            else:
                print(f"⚠️ No labels found for model '{model_name}'")
                return {}

            # Convert numpy types safely
            unique, counts = np.unique(labels, return_counts=True)
            distribution = {str(int(k)): int(v) for k, v in zip(unique, counts)}

            return distribution

        except Exception as e:
            print(f"⚠️ Error in get_cluster_distribution for {model_name}: {e}")
            return {}

    # ---------------- Train All ---------------- #
    def train_all_models(self, X, n_clusters=2):
        self.X_train = X  # Save training data for prediction
        results = {}
        
        print("Training KMeans...")
        _, labels = self.train_kmeans(X, n_clusters=n_clusters)
        results['kmeans'] = self.evaluate_clustering(X, labels, 'kmeans')
        
        print("Training DBSCAN...")
        _, labels = self.train_dbscan(X, eps=0.5, min_samples=5)
        results['dbscan'] = self.evaluate_clustering(X, labels, 'dbscan')
        
        print("Training Hierarchical...")
        _, labels = self.train_hierarchical(X, n_clusters=n_clusters)
        results['hierarchical'] = self.evaluate_clustering(X, labels, 'hierarchical')
        
        print("Training Fuzzy C-Means...")
        _, _, labels = self.train_fuzzy_cmeans(X, n_clusters=n_clusters)
        results['fuzzy_cmeans'] = self.evaluate_clustering(X, labels, 'fuzzy_cmeans')
        
        print("Training GMM...")
        _, labels = self.train_gmm(X, n_components=n_clusters)
        results['gmm'] = self.evaluate_clustering(X, labels, 'gmm')
        
        print("Training Autoencoder...")
        _, _, labels = self.train_autoencoder(X, encoding_dim=8, epochs=50)
        results['autoencoder'] = self.evaluate_clustering(X, labels, 'autoencoder')
        
        return results

    # ---------------- Prediction ---------------- #
    def predict_cluster(self, model_name, X):
        X = np.array(X).reshape(1, -1)
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        model = self.trained_models[model_name]

        if model_name in ['kmeans', 'gmm']:
            return model.predict(X)
        elif model_name == 'dbscan':
            core_samples = self.X_train[model.core_sample_indices_]
            core_labels = model.labels_[model.core_sample_indices_]
            nn = NearestNeighbors(n_neighbors=1).fit(core_samples)
            dist, idx = nn.kneighbors(X)
            labels = [int(core_labels[i]) if dist[i][0] <= model.eps else -1 for i in range(len(X))]
            return np.array(labels)
        elif model_name == 'hierarchical':
            labels_train = model.fit_predict(self.X_train)
            centroids = np.array([self.X_train[labels_train == i].mean(axis=0) for i in np.unique(labels_train)])
            labels, _ = pairwise_distances_argmin_min(X, centroids)
            return labels
        elif model_name == 'fuzzy_cmeans':
            X_transposed = X.T
            u, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
                X_transposed, model['centers'], model['m'], error=0.005, maxiter=1000
            )
            return np.argmax(u, axis=0)
        elif model_name == 'autoencoder':
            encoded = model['encoder'].predict(X, verbose=0)
            return model['kmeans'].predict(encoded)

    # ---------------- Comparison ---------------- #
    def compare_models(self):
        comparison = {}
        for model_name, metrics in self.model_results.items():
            comparison[model_name] = {
                'silhouette_score': metrics['silhouette_score'],
                'davies_bouldin_score': metrics['davies_bouldin_score'],
                'calinski_harabasz_score': metrics['calinski_harabasz_score'],
                'n_clusters': metrics['n_clusters']
            }
        return comparison

    def get_best_model(self, metric='silhouette_score'):
        if not self.model_results:
            return None
        if metric == 'davies_bouldin_score':
            best_model = min(self.model_results.items(), key=lambda x: x[1][metric])
        else:
            best_model = max(self.model_results.items(), key=lambda x: x[1][metric])
        return best_model[0], best_model[1][metric]

    # ---------------- Save Models ---------------- #
    def save_model(self, model_name, filepath):
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        # Handle different model types
        if model_name == 'autoencoder':
            # Save autoencoder components separately
            model['autoencoder'].save(filepath.replace('.pkl', '_autoencoder.h5'))
            model['encoder'].save(filepath.replace('.pkl', '_encoder.h5'))
            joblib.dump(model['kmeans'], filepath)
        elif model_name == 'fuzzy_cmeans':
            # Save fuzzy c-means parameters
            joblib.dump(model, filepath)
        else:
            joblib.dump(model, filepath)
        
        print(f"✅ Saved {model_name} to {filepath}")

    # ---------------- Get All Cluster Distributions ---------------- #
    def get_all_cluster_distributions(self):
        """Get cluster distributions for all trained models"""
        distributions = {}
        for model_name in self.trained_models.keys():
            distributions[model_name] = self.get_cluster_distribution(model_name)
        return distributions


if __name__ == "__main__":
    import os
    import json
    import numpy as np
    import pandas as pd
    from data_preprocessing import DataPreprocessor

    print("\n=== Running Unsupervised Models ===\n")

    # Load and preprocess dataset
    print("📂 Loading dataset from heart.csv...")
    df = pd.read_csv("heart.csv")
    print(f"✅ Dataset loaded with shape: {df.shape}\n")

    preprocessor = DataPreprocessor()
    X, _, _, _ = preprocessor.prepare_data(df)

    models = UnsupervisedModels()
    print("🚀 Training all unsupervised models...")
    results = models.train_all_models(X, n_clusters=2)

    print("\n--- Clustering Evaluation Results ---\n")
    for name, metrics in results.items():
        print(f"Model: {name}")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print("----------------------------------------")

    best_model, best_score = models.get_best_model(metric='silhouette_score')
    print(f"\n🏆 Best Model: {best_model} (Silhouette Score: {best_score:.4f})")

    # Get cluster distributions
    print("\n--- Cluster Distributions ---")
    distributions = models.get_all_cluster_distributions()
    for model_name, distribution in distributions.items():
        print(f"{model_name}: {distribution}")

    print("\n💾 Saving trained unsupervised models...\n")
    os.makedirs("models", exist_ok=True)
    for model_name in models.trained_models.keys():
        filepath = os.path.join("models", f"{model_name}.pkl")
        models.save_model(model_name, filepath)

    print("\n🎉 All unsupervised models saved successfully in the 'models/' folder.\n")

    # ✅ Save unsupervised results
    def convert_to_python(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)

    # Save results with cluster distributions
    full_results = {
        'evaluation_metrics': results,
        'cluster_distributions': distributions
    }
    
    with open("unsupervised_results.json", "w") as f:
        json.dump(full_results, f, indent=4, default=convert_to_python)

    print("✅ unsupervised_results.json created successfully!")