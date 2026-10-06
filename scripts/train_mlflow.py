import mlflow
import mlflow.sklearn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from src.data.load_features import load_features
from src.features.prepare import prepare_features


# Locked threshold selected during validation
FINAL_THRESHOLD = 0.32


def main():

    # --------------------------------------------------
    # 1. Set MLflow experiment
    # --------------------------------------------------

    mlflow.set_experiment(
        "Customer Churn Prediction"
    )

    # --------------------------------------------------
    # 2. Load feature data
    # --------------------------------------------------

    df = load_features()

    X, y, preprocessor = prepare_features(df)

    # --------------------------------------------------
    # 3. Train/Test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------
    # 4. Start MLflow run
    # --------------------------------------------------

    with mlflow.start_run(
        run_name="Final Tuned XGBoost"
    ):

        # --------------------------------------------------
        # 5. Model configuration
        # --------------------------------------------------

        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )

        pipeline = Pipeline(
            [
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "classifier",
                    model,
                ),
            ]
        )

        # --------------------------------------------------
        # 6. Log parameters
        # --------------------------------------------------

        mlflow.log_params(
            {
                "model": "XGBoost",
                "n_estimators": 200,
                "learning_rate": 0.03,
                "max_depth": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "decision_threshold": FINAL_THRESHOLD,
            }
        )

        # --------------------------------------------------
        # 7. Train
        # --------------------------------------------------

        pipeline.fit(
            X_train,
            y_train,
        )

        # --------------------------------------------------
        # 8. Predictions
        # --------------------------------------------------

        y_prob = pipeline.predict_proba(
            X_test
        )[:, 1]

        y_pred = (
            y_prob >= FINAL_THRESHOLD
        ).astype(int)

        # --------------------------------------------------
        # 9. Calculate metrics
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
        # 10. Log metrics
        # --------------------------------------------------

        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }
        )

        # --------------------------------------------------
        # 11. Print results
        # --------------------------------------------------

        print("\n" + "=" * 50)
        print("MLFLOW XGBOOST RUN")
        print("=" * 50)

        print(
            f"Threshold : {FINAL_THRESHOLD:.2f}"
        )

        print(
            f"Accuracy  : {accuracy:.4f}"
        )

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1 Score  : {f1:.4f}"
        )

        print(
            f"ROC-AUC   : {roc_auc:.4f}"
        )

        print(
            f"PR-AUC    : {pr_auc:.4f}"
        )

        print("\n--- CONFUSION MATRIX ---")

        print(
            confusion_matrix(
                y_test,
                y_pred,
            )
        )

        # --------------------------------------------------
        # 12. Log model
        # --------------------------------------------------

        mlflow.sklearn.log_model(
            pipeline,
            name="churn_pipeline",
            skops_trusted_types=[
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )

        print("\nComplete pipeline logged to MLflow successfully.")
        print("Run ID:", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()