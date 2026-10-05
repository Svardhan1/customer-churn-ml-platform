from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.load_features import load_features
from src.features.prepare import prepare_features


# Locked during validation
FINAL_THRESHOLD = 0.32


def main():

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------

    df = load_features()

    X, y, preprocessor = prepare_features(df)

    # --------------------------------------------------
    # 2. Recreate the exact train/test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------
    # 3. Final tuned XGBoost configuration
    # --------------------------------------------------

    pipeline = Pipeline(
        [
            (
                "preprocessor",
                preprocessor,
            ),
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

    # --------------------------------------------------
    # 4. Train on training data
    # --------------------------------------------------

    pipeline.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------
    # 5. Get probability predictions
    # --------------------------------------------------

    y_prob = pipeline.predict_proba(
        X_test
    )[:, 1]

    # --------------------------------------------------
    # 6. Apply LOCKED threshold
    # --------------------------------------------------

    y_pred = (
        y_prob >= FINAL_THRESHOLD
    ).astype(int)

    # --------------------------------------------------
    # 7. Calculate final metrics
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob,
    )

    pr_auc = average_precision_score(
        y_test,
        y_prob,
    )

    # --------------------------------------------------
    # 8. Print results
    # --------------------------------------------------

    print("\n" + "=" * 50)
    print("FINAL XGBOOST TEST EVALUATION")
    print("=" * 50)

    print(
        f"Decision Threshold : {FINAL_THRESHOLD:.2f}"
    )

    print(
        f"Accuracy           : {accuracy:.4f}"
    )

    print(
        f"Precision          : {precision:.4f}"
    )

    print(
        f"Recall             : {recall:.4f}"
    )

    print(
        f"F1 Score           : {f1:.4f}"
    )

    print(
        f"ROC-AUC            : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC             : {pr_auc:.4f}"
    )

    # --------------------------------------------------
    # 9. Confusion matrix
    # --------------------------------------------------

    print("\n--- CONFUSION MATRIX ---")

    print(
        confusion_matrix(
            y_test,
            y_pred,
        )
    )

    # --------------------------------------------------
    # 10. Classification report
    # --------------------------------------------------

    print("\n--- CLASSIFICATION REPORT ---")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Churn",
                "Churn",
            ],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()