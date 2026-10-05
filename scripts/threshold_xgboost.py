from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.load_features import load_features
from src.features.prepare import prepare_features


def main():
    # Load data
    df = load_features()

    # Prepare features
    X, y, preprocessor = prepare_features(df)

    # Same train/test split used in previous experiments
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Tuned XGBoost model
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.03,
                    max_depth=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Get churn probabilities
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Threshold analysis
    results = []

    for threshold in [i / 100 for i in range(10, 91)]:
        y_pred = (y_prob >= threshold).astype(int)

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    # Best F1
    best_f1 = max(results, key=lambda x: x["f1"])

    # Best recall while maintaining at least 50% precision
    valid_recall = [
        result
        for result in results
        if result["precision"] >= 0.50
    ]

    best_recall = max(valid_recall, key=lambda x: x["recall"])

    print("\n--- BEST F1 THRESHOLD ---")
    print(f"Threshold : {best_f1['threshold']:.2f}")
    print(f"Precision : {best_f1['precision']:.4f}")
    print(f"Recall    : {best_f1['recall']:.4f}")
    print(f"F1 Score  : {best_f1['f1']:.4f}")

    print("\n--- BEST RECALL WITH PRECISION >= 0.50 ---")
    print(f"Threshold : {best_recall['threshold']:.2f}")
    print(f"Precision : {best_recall['precision']:.4f}")
    print(f"Recall    : {best_recall['recall']:.4f}")
    print(f"F1 Score  : {best_recall['f1']:.4f}")

    # Confusion matrix for best F1 threshold
    best_threshold = best_f1["threshold"]

    y_best = (y_prob >= best_threshold).astype(int)

    print("\n--- CONFUSION MATRIX AT BEST F1 THRESHOLD ---")
    print(confusion_matrix(y_test, y_best))

    print("\n--- THRESHOLD ANALYSIS ---")

    for result in results:
        if result["threshold"] in [
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
        ]:
            print(
                f"Threshold={result['threshold']:.2f} | "
                f"Precision={result['precision']:.4f} | "
                f"Recall={result['recall']:.4f} | "
                f"F1={result['f1']:.4f}"
            )


if __name__ == "__main__":
    main()