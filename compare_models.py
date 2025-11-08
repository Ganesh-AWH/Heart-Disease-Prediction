from supervised_models import SupervisedModels
from unsupervised_models import UnsupervisedModels
from data_preprocessing import DataPreprocessor
from model_comparison import ModelComparison
import pandas as pd

print("\n================================================================================")
print(" Heart Disease Prediction - Supervised vs Unsupervised Model Comparison")
print("================================================================================\n")

# Step 1: Load and preprocess dataset
df = pd.read_csv("heart.csv")
preprocessor = DataPreprocessor()

print("🔹 Loading and preprocessing data...")
X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)

# Step 2: Train supervised models
print("\n=== Training Supervised Models ===")
supervised = SupervisedModels()
supervised_results = supervised.train_all_models(X_train, y_train, X_test, y_test)

# Step 3: Train unsupervised models
print("\n=== Training Unsupervised Models ===")
unsupervised = UnsupervisedModels()
unsupervised_results = unsupervised.train_all_models(X_train)

# Step 4: Compare results
print("\n=== Comparing Models ===")
comparator = ModelComparison()
report = comparator.generate_comprehensive_report(
    supervised_results=supervised_results,
    unsupervised_results=unsupervised_results
)

# Step 5: Display results summary
print("\n--- Supervised Model Summary ---")
print("Best Accuracy Model:", report['supervised_learning']['best_accuracy'])
print("Average Accuracy:", report['supervised_learning']['summary']['avg_accuracy'])
print("Average F1:", report['supervised_learning']['summary']['avg_f1'])

print("\n--- Unsupervised Model Summary ---")
print("Best Silhouette Model:", report['unsupervised_learning']['best_silhouette'])
print("Average Silhouette:", report['unsupervised_learning']['summary']['avg_silhouette'])

print("\n🏆 Recommendation:")
print(f"Best Supervised → {report['recommendations']['best_supervised_model']}")
print(f"Best Unsupervised → {report['recommendations']['best_unsupervised_model']}")

# Step 6: Export report
output_file = "model_comparison_report.json"
comparator.export_results(output_file)
print(f"\n📁 Report saved as: {output_file}")


# ============================================
# Step 7: Visualization (Save as PNG)
# ============================================
import matplotlib.pyplot as plt
import numpy as np
import json
import os

print("\n📊 Generating visualization...")

# Create visualization folder
os.makedirs("visualizations", exist_ok=True)

# --- Load results (already available in memory, but safer to reload if needed) ---
supervised_file = "supervised_results.json"
unsupervised_file = "unsupervised_results.json"

with open(supervised_file, "r") as f:
    supervised_data = json.load(f)

with open(unsupervised_file, "r") as f:
    unsupervised_data = json.load(f)

# --- Prepare supervised data ---
supervised_models = list(supervised_data.keys())
supervised_acc = [v["accuracy"] for v in supervised_data.values()]
supervised_f1 = [v["f1_score"] for v in supervised_data.values()]

# --- Prepare unsupervised data ---
unsupervised_models = list(unsupervised_data.keys())
unsupervised_sil = [v["silhouette_score"] for v in unsupervised_data.values()]
unsupervised_db = [v["davies_bouldin_score"] for v in unsupervised_data.values()]

# --- Plot comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Supervised plot
axes[0].bar(supervised_models, supervised_acc, color='royalblue', alpha=0.7, label='Accuracy')
axes[0].bar(supervised_models, supervised_f1, color='orange', alpha=0.7, label='F1-Score')
axes[0].set_title("Supervised Learning Models")
axes[0].set_ylabel("Score")
axes[0].legend()
axes[0].tick_params(axis='x', rotation=45)

# Unsupervised plot
axes[1].bar(unsupervised_models, unsupervised_sil, color='seagreen', alpha=0.7, label='Silhouette Score')
axes[1].bar(unsupervised_models, unsupervised_db, color='crimson', alpha=0.5, label='Davies-Bouldin (lower better)')
axes[1].set_title("Unsupervised Learning Models")
axes[1].set_ylabel("Score")
axes[1].legend()
axes[1].tick_params(axis='x', rotation=45)

plt.suptitle("📈 Model Performance Comparison: Supervised vs Unsupervised", fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save as PNG
plt.savefig("visualizations/model_comparison.png", dpi=300)
plt.close()

print("✅ Visualization saved as: visualizations/model_comparison.png")
