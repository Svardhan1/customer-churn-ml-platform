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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.load_features import load_features
from src.features.prepare import prepare_features


def main():

    # 1. Load data
    df = load_features()

    # 2. Prepare features
    X, y, preprocessor = prepare_features(df)

    # 3. Keep final test set untouched
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # 4. Build XGBoost pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )

    # 5. Controlled hyperparameter search
    param_grid = {
        "classifier__n_estimators": [200, 300],
        "classifier__learning_rate": [0.03, 0.05],
        "classifier__max_depth": [3, 4],
        "classifier__subsample": [0.8],
        "classifier__colsample_bytree": [0.8],
    }

    # 6. Grid search with 5-fold CV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=1,
        verbose=1,
    )

    # 7. Train
    grid_search.fit(X_train, y_train)

    print("\n--- BEST PARAMETERS ---")
    print(grid_search.best_params_)

    print("\n--- BEST CROSS-VALIDATION ROC-AUC ---")
    print(f"{grid_search.best_score_:.4f}")

    # 8. Final evaluation on untouched test set
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print("\n--- FINAL TEST EVALUATION ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    print("\n--- CONFUSION MATRIX ---")
    print(confusion_matrix(y_test, y_pred))

    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()