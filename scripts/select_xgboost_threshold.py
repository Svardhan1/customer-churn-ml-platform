import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.load_features import load_features
from src.features.prepare import prepare_features


def main():

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------

    df = load_features()

    X, y, preprocessor = prepare_features(df)

    # --------------------------------------------------
    # 2. Keep final test set untouched
    # --------------------------------------------------

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # --------------------------------------------------
    # 3. Tuned XGBoost model
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
    # 4. Generate out-of-fold probabilities
    # --------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    y_train_prob = cross_val_predict(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    print("\n--- OUT-OF-FOLD VALIDATION ---")

    print(
        "Validation ROC-AUC:",
        round(
            roc_auc_score(
                y_train,
                y_train_prob,
            ),
            4,
        ),
    )

    # --------------------------------------------------
    # 5. Find best threshold
    # --------------------------------------------------

    results = []

    for threshold in np.arange(
        0.10,
        0.91,
        0.01,
    ):

        y_pred = (
            y_train_prob >= threshold
        ).astype(int)

        precision = precision_score(
            y_train,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_train,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_train,
            y_pred,
            zero_division=0,
        )

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    # --------------------------------------------------
    # 6. Best F1 threshold
    # --------------------------------------------------

    best_f1 = max(
        results,
        key=lambda x: x["f1"],
    )

    print("\n--- BEST VALIDATION F1 THRESHOLD ---")

    print(
        f"Threshold : {best_f1['threshold']:.2f}"
    )

    print(
        f"Precision : {best_f1['precision']:.4f}"
    )

    print(
        f"Recall    : {best_f1['recall']:.4f}"
    )

    print(
        f"F1 Score  : {best_f1['f1']:.4f}"
    )

    # --------------------------------------------------
    # 7. Best recall with precision >= 0.50
    # --------------------------------------------------

    valid_results = [
        result
        for result in results
        if result["precision"] >= 0.50
    ]

    best_recall = max(
        valid_results,
        key=lambda x: x["recall"],
    )

    print(
        "\n--- BEST VALIDATION RECALL "
        "WITH PRECISION >= 0.50 ---"
    )

    print(
        f"Threshold : {best_recall['threshold']:.2f}"
    )

    print(
        f"Precision : {best_recall['precision']:.4f}"
    )

    print(
        f"Recall    : {best_recall['recall']:.4f}"
    )

    print(
        f"F1 Score  : {best_recall['f1']:.4f}"
    )

    # --------------------------------------------------
    # 8. Show useful thresholds
    # --------------------------------------------------

    print("\n--- VALIDATION THRESHOLD ANALYSIS ---")

    for result in results:

        if round(result["threshold"], 2) in [
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
        ]:

            print(
                f"Threshold={result['threshold']:.2f} | "
                f"Precision={result['precision']:.4f} | "
                f"Recall={result['recall']:.4f} | "
                f"F1={result['f1']:.4f}"
            )


if __name__ == "__main__":
    main()