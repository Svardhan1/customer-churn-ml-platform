import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.data.load_features import load_features
from src.features.prepare import prepare_features


def main():
    # 1. Load data
    df = load_features()

    # 2. Prepare features
    X, y, preprocessor = prepare_features(df)

    # 3. Same train/test split used in model training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 4. Build Logistic Regression pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000),
            ),
        ]
    )

    # 5. Train
    model.fit(X_train, y_train)

    # 6. Get churn probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # 7. Test different thresholds
    thresholds = np.arange(0.10, 0.91, 0.01)

    results = []

    for threshold in thresholds:

        y_pred = (y_prob >= threshold).astype(int)

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

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    results_df = pd.DataFrame(results)
    best_f1_row = results_df.loc[
    results_df["f1"].idxmax()]

    print("\n--- BEST F1 THRESHOLD ---")

    print(
        f"Threshold : {best_f1_row['threshold']:.2f}"
    )

    print(
        f"Precision : {best_f1_row['precision']:.4f}"
    )

    print(
        f"Recall    : {best_f1_row['recall']:.4f}"
    )

    print(
        f"F1 Score  : {best_f1_row['f1']:.4f}"
    )
    
    eligible = results_df[
    results_df["precision"] >= 0.50]
    
    best_recall_row = eligible.loc[
    eligible["recall"].idxmax()]
    
    print("\n--- BEST RECALL WITH PRECISION >= 50% ---")

    print(
        f"Threshold : {best_recall_row['threshold']:.2f}"
    )

    print(
        f"Precision : {best_recall_row['precision']:.4f}"
    )

    print(
        f"Recall    : {best_recall_row['recall']:.4f}"
    )

    print(
        f"F1 Score  : {best_recall_row['f1']:.4f}"
    )


if __name__ == "__main__":
    main()