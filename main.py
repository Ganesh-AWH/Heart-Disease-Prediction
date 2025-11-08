import sys
from data_preprocessing import DataPreprocessor
from supervised_models import SupervisedModels
from unsupervised_models import UnsupervisedModels
from model_comparison import ModelComparison


def main():
    print("=" * 80)
    print("Heart Disease Prediction - Comparative Analysis")
    print("Supervised vs Unsupervised Learning Algorithms")
    print("=" * 80)
    print()

    preprocessor = DataPreprocessor()
    supervised_models = SupervisedModels()
    unsupervised_models = UnsupervisedModels()
    model_comparison = ModelComparison()

    print("Step 1: Loading and Preprocessing Data...")
    df = preprocessor.load_data(filepath="heart.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    print()

    print("Step 2: Preparing Data for Supervised Learning...")
    X_train_scaled, X_test_scaled, y_train, y_test = preprocessor.prepare_data(df)
    print(f"Training set size: {X_train_scaled.shape[0]}")
    print(f"Test set size: {X_test_scaled.shape[0]}")
    print()

    print("=" * 80)
    print("SUPERVISED LEARNING - Training Models")
    print("=" * 80)
    print()

    supervised_results = supervised_models.train_all_models(
        X_train_scaled, y_train,
        X_test_scaled, y_test
    )

    print("\nSupervised Learning Results:")
    print("-" * 80)
    for model_name, metrics in supervised_results.items():
        print(f"\n{model_name.upper()}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")

    best_sup_model, best_sup_score = supervised_models.get_best_model('accuracy')
    print(f"\n✓ Best Supervised Model: {best_sup_model} (Accuracy: {best_sup_score:.4f})")
    print()

    print("=" * 80)
    print("UNSUPERVISED LEARNING - Training Models")
    print("=" * 80)
    print()


    print("Step 3: Preparing Data for Unsupervised Learning...")
    df_unsupervised = df.copy()
    X_unsupervised = preprocessor.prepare_unsupervised_data(df_unsupervised)
    X_unsupervised = X_unsupervised  # keep consistent naming


    unsupervised_results = unsupervised_models.train_all_models(X_unsupervised, n_clusters=2)

    print("\nUnsupervised Learning Results:")
    print("-" * 80)
    for model_name, metrics in unsupervised_results.items():
        print(f"\n{model_name.upper()}")
        print(f"  Silhouette Score:       {metrics['silhouette_score']:.4f}")
        print(f"  Davies-Bouldin Index:   {metrics['davies_bouldin_score']:.4f}")
        print(f"  Calinski-Harabasz:      {metrics['calinski_harabasz_score']:.4f}")
        print(f"  Number of Clusters:     {metrics['n_clusters']}")
        if metrics['n_noise_points'] > 0:
            print(f"  Noise Points:           {metrics['n_noise_points']}")

    best_unsup_model, best_unsup_score = unsupervised_models.get_best_model('silhouette_score')
    print(f"\n✓ Best Unsupervised Model: {best_unsup_model} (Silhouette Score: {best_unsup_score:.4f})")
    print()

    print("=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)
    print()

    report = model_comparison.generate_comprehensive_report(
        supervised_results,
        unsupervised_results
    )

    print("Summary of Findings:")
    print("-" * 80)
    print("\nSupervised Learning:")
    sup_summary = report['supervised_learning']['summary']
    print(f"  Average Accuracy:  {sup_summary['avg_accuracy']:.4f}")
    print(f"  Average Precision: {sup_summary['avg_precision']:.4f}")
    print(f"  Average Recall:    {sup_summary['avg_recall']:.4f}")
    print(f"  Average F1-Score:  {sup_summary['avg_f1']:.4f}")

    print("\nUnsupervised Learning:")
    unsup_summary = report['unsupervised_learning']['summary']
    print(f"  Average Silhouette Score:      {unsup_summary['avg_silhouette']:.4f}")
    print(f"  Average Davies-Bouldin Index:  {unsup_summary['avg_davies_bouldin']:.4f}")
    print(f"  Average Calinski-Harabasz:     {unsup_summary['avg_calinski']:.4f}")

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    recommendations = report['recommendations']

    print(f"\nBest Supervised Model: {recommendations['best_supervised_model']}")
    print(f"Best Unsupervised Model: {recommendations['best_unsupervised_model']}")

    print("\nWhen to Use Supervised Learning:")
    for use_case in recommendations['use_cases']['supervised']:
        print(f"  • {use_case}")

    print("\nWhen to Use Unsupervised Learning:")
    for use_case in recommendations['use_cases']['unsupervised']:
        print(f"  • {use_case}")

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)

    export_path = model_comparison.export_results('model_comparison_report.json')
    print(f"\nDetailed report exported to: {export_path}")

    return report


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)
