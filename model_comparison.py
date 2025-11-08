import numpy as np
import pandas as pd
import json
from datetime import datetime


class ModelComparison:
    def __init__(self):
        self.comparison_results = {}

    def compare_supervised_models(self, supervised_results):
        comparison_df = pd.DataFrame({
            'Model': list(supervised_results.keys()),
            'Accuracy': [results['accuracy'] for results in supervised_results.values()],
            'Precision': [results['precision'] for results in supervised_results.values()],
            'Recall': [results['recall'] for results in supervised_results.values()],
            'F1-Score': [results['f1_score'] for results in supervised_results.values()]
        })

        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

        self.comparison_results['supervised'] = {
            'comparison_table': comparison_df.to_dict('records'),
            'best_accuracy': comparison_df.iloc[0]['Model'],
            'best_precision': comparison_df.sort_values('Precision', ascending=False).iloc[0]['Model'],
            'best_recall': comparison_df.sort_values('Recall', ascending=False).iloc[0]['Model'],
            'best_f1': comparison_df.sort_values('F1-Score', ascending=False).iloc[0]['Model'],
            'summary': {
                'avg_accuracy': float(comparison_df['Accuracy'].mean()),
                'avg_precision': float(comparison_df['Precision'].mean()),
                'avg_recall': float(comparison_df['Recall'].mean()),
                'avg_f1': float(comparison_df['F1-Score'].mean())
            }
        }

        return self.comparison_results['supervised']

    def compare_unsupervised_models(self, unsupervised_results):
        comparison_df = pd.DataFrame({
            'Model': list(unsupervised_results.keys()),
            'Silhouette Score': [results['silhouette_score'] for results in unsupervised_results.values()],
            'Davies-Bouldin Index': [results['davies_bouldin_score'] for results in unsupervised_results.values()],
            'Calinski-Harabasz Score': [results['calinski_harabasz_score'] for results in unsupervised_results.values()],
            'N Clusters': [results['n_clusters'] for results in unsupervised_results.values()]
        })

        comparison_df = comparison_df.sort_values('Silhouette Score', ascending=False)

        self.comparison_results['unsupervised'] = {
            'comparison_table': comparison_df.to_dict('records'),
            'best_silhouette': comparison_df.iloc[0]['Model'],
            'best_davies_bouldin': comparison_df.sort_values('Davies-Bouldin Index', ascending=True).iloc[0]['Model'],
            'best_calinski': comparison_df.sort_values('Calinski-Harabasz Score', ascending=False).iloc[0]['Model'],
            'summary': {
                'avg_silhouette': float(comparison_df['Silhouette Score'].mean()),
                'avg_davies_bouldin': float(comparison_df['Davies-Bouldin Index'].replace([np.inf, -np.inf], np.nan).mean()),
                'avg_calinski': float(comparison_df['Calinski-Harabasz Score'].mean())
            }
        }

        return self.comparison_results['unsupervised']

    def generate_comprehensive_report(self, supervised_results=None, unsupervised_results=None):
        report = {
            'timestamp': datetime.now().isoformat(),
            'report_type': 'Heart Disease Prediction - Model Comparison'
        }

        if supervised_results:
            report['supervised_learning'] = self.compare_supervised_models(supervised_results)

        if unsupervised_results:
            report['unsupervised_learning'] = self.compare_unsupervised_models(unsupervised_results)

        if supervised_results and unsupervised_results:
            report['recommendations'] = self.generate_recommendations(
                supervised_results,
                unsupervised_results
            )

        return report

    def generate_recommendations(self, supervised_results, unsupervised_results):
        sup_comparison = self.comparison_results.get('supervised', {})
        unsup_comparison = self.comparison_results.get('unsupervised', {})

        recommendations = {
            'best_supervised_model': sup_comparison.get('best_accuracy', 'N/A'),
            'best_unsupervised_model': unsup_comparison.get('best_silhouette', 'N/A'),
            'use_cases': {
                'supervised': [
                    'When labeled data is available',
                    'For accurate binary classification (disease vs no disease)',
                    'When interpretability is important (Decision Trees, Logistic Regression)',
                    'For real-time predictions with high accuracy'
                ],
                'unsupervised': [
                    'When labeled data is not available',
                    'For exploratory data analysis',
                    'To identify patient subgroups and risk patterns',
                    'For anomaly detection in patient data',
                    'To reduce dimensionality for visualization'
                ]
            },
            'performance_summary': {
                'supervised_avg_accuracy': sup_comparison.get('summary', {}).get('avg_accuracy', 0),
                'unsupervised_avg_silhouette': unsup_comparison.get('summary', {}).get('avg_silhouette', 0)
            }
        }

        return recommendations

    def export_results(self, filepath='model_comparison_report.json'):
        if not self.comparison_results:
            raise ValueError("No comparison results available. Run comparison first.")

        with open(filepath, 'w') as f:
            json.dump(self.comparison_results, f, indent=4)

        return filepath

    def get_model_rankings(self, model_type='supervised'):
        if model_type not in self.comparison_results:
            return None

        comparison = self.comparison_results[model_type]

        if model_type == 'supervised':
            rankings = {
                'by_accuracy': sorted(
                    comparison['comparison_table'],
                    key=lambda x: x['Accuracy'],
                    reverse=True
                ),
                'by_f1_score': sorted(
                    comparison['comparison_table'],
                    key=lambda x: x['F1-Score'],
                    reverse=True
                )
            }
        else:
            rankings = {
                'by_silhouette': sorted(
                    comparison['comparison_table'],
                    key=lambda x: x['Silhouette Score'],
                    reverse=True
                ),
                'by_davies_bouldin': sorted(
                    comparison['comparison_table'],
                    key=lambda x: x['Davies-Bouldin Index'],
                    reverse=False
                )
            }

        return rankings

    def calculate_statistical_significance(self, results_list):
        if len(results_list) < 2:
            return None

        accuracies = [r['accuracy'] for r in results_list]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        return {
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'min_accuracy': float(np.min(accuracies)),
            'max_accuracy': float(np.max(accuracies)),
            'variance': float(np.var(accuracies))
        }
if __name__ == "__main__":
    from joblib import load

    # --- Step 1: Load your trained models and scalers ---
    # Adjust paths as needed
    models = {
        'Random Forest': load('models/random_forest.joblib'),
        'SVM': load('models/svm.joblib'),
        'Decision Tree': load('models/decision_tree.joblib'),
        'Logistic Regression': load('models/logistic_regression.joblib')
        # add other models
    }

    unsupervised_models = {
        'KMeans': load('models/kmeans.joblib'),
        'DBSCAN': load('models/dbscan.joblib'),
        'Hierarchical': load('models/hierarchical.joblib')
        # add other unsupervised models
    }

    # --- Step 2: Collect real results ---
    # Supervised results
    supervised_results = {}
    for name, model in models.items():
        # Assuming you have a function evaluate_supervised_model returning metrics
        metrics = evaluate_supervised_model(model)  # should return dict with keys: accuracy, precision, recall, f1_score
        supervised_results[name] = metrics

    # Unsupervised results
    unsupervised_results = {}
    for name, model in unsupervised_models.items():
        # Assuming you have a function evaluate_unsupervised_model returning metrics
        metrics = evaluate_unsupervised_model(model)  # should return dict with keys: silhouette_score, davies_bouldin_score, calinski_harabasz_score, n_clusters
        unsupervised_results[name] = metrics

    # --- Step 3: Initialize ModelComparison ---
    comparator = ModelComparison()

    # --- Step 4: Generate dynamic report ---
    report = comparator.generate_comprehensive_report(
        supervised_results=supervised_results,
        unsupervised_results=unsupervised_results
    )

    # --- Step 5: Print and export report ---
    print(json.dumps(report, indent=4))
    comparator.export_results('model_comparison_report.json')

    # --- Step 6: Show rankings dynamically ---
    sup_rankings = comparator.get_model_rankings('supervised')
    unsup_rankings = comparator.get_model_rankings('unsupervised')

    print("\nTop Supervised Models by Accuracy:")
    for i, model in enumerate(sup_rankings['by_accuracy'], start=1):
        print(f"{i}. {model['Model']} - Accuracy: {model['Accuracy']}")

    print("\nTop Unsupervised Models by Silhouette Score:")
    for i, model in enumerate(unsup_rankings['by_silhouette'], start=1):
        print(f"{i}. {model['Model']} - Silhouette Score: {model['Silhouette Score']}")
