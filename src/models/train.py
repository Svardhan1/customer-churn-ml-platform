import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.data.load_features import load_features
from src.features.prepare import prepare_features
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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

def main():
    # 1. Load feature data from PostgreSQL
    df = load_features()

    # 2. Separate features and target
    X, y, preprocessor = prepare_features(df)

    # 3. Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # 4. Build ML pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000
                ),
            ),
        ]
    )

    # 5. Train model
    model.fit(X_train, y_train)

    print("\nModel training completed successfully.")

# 6. Generate predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

# 7. Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

# 8. Display metrics
    print("\n--- MODEL EVALUATION ---")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    
    # 9. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- CONFUSION MATRIX ---")
    print(cm)

# 10. Classification report
    print("\n--- CLASSIFICATION REPORT ---")
    print(classification_report(y_test, y_pred))
    
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}") 
    
    
    # 11. Random Forest model
    random_forest = Pipeline(
        steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
    )

# 12. Train Random Forest
    random_forest.fit(X_train, y_train)

    print("\nRandom Forest training completed successfully.")

# 13. Generate predictions
    rf_pred = random_forest.predict(X_test)
    rf_prob = random_forest.predict_proba(X_test)[:, 1]

# 14. Calculate metrics
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_precision = precision_score(y_test, rf_pred)
    rf_recall = recall_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred)
    rf_roc_auc = roc_auc_score(y_test, rf_prob)
    rf_pr_auc = average_precision_score(y_test, rf_prob)

    print("\n--- RANDOM FOREST EVALUATION ---")
    print(f"Accuracy : {rf_accuracy:.4f}")
    print(f"Precision: {rf_precision:.4f}")
    print(f"Recall   : {rf_recall:.4f}")
    print(f"F1 Score : {rf_f1:.4f}")
    print(f"ROC-AUC  : {rf_roc_auc:.4f}")
    print(f"PR-AUC   : {rf_pr_auc:.4f}")

# 15. Random Forest confusion matrix
    rf_cm = confusion_matrix(y_test, rf_pred)

    print("\n--- RANDOM FOREST CONFUSION MATRIX ---")
    print(rf_cm)

# 16. Random Forest classification report
    print("\n--- RANDOM FOREST CLASSIFICATION REPORT ---")
    print(classification_report(y_test, rf_pred))
    
    
    # 17. XGBoost model
    xgb_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # 18. Train XGBoost
    xgb_model.fit(X_train, y_train)

    print("\nXGBoost training completed successfully.")

    # 19. Generate predictions
    xgb_pred = xgb_model.predict(X_test)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

    # 20. Calculate metrics
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    xgb_precision = precision_score(y_test, xgb_pred)
    xgb_recall = recall_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred)
    xgb_roc_auc = roc_auc_score(y_test, xgb_prob)
    xgb_pr_auc = average_precision_score(y_test, xgb_prob)

    print("\n--- XGBOOST EVALUATION ---")
    print(f"Accuracy : {xgb_accuracy:.4f}")
    print(f"Precision: {xgb_precision:.4f}")
    print(f"Recall   : {xgb_recall:.4f}")
    print(f"F1 Score : {xgb_f1:.4f}")
    print(f"ROC-AUC  : {xgb_roc_auc:.4f}")
    print(f"PR-AUC   : {xgb_pr_auc:.4f}")

    # 21. XGBoost confusion matrix
    xgb_cm = confusion_matrix(y_test, xgb_pred)

    print("\n--- XGBOOST CONFUSION MATRIX ---")
    print(xgb_cm)

    # 22. XGBoost classification report
    print("\n--- XGBOOST CLASSIFICATION REPORT ---")
    print(classification_report(y_test, xgb_pred))
        
if __name__ == "__main__":
    main()