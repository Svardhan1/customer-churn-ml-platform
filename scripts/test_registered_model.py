import mlflow.sklearn

from src.data.load_features import load_features
from src.features.prepare import prepare_features


MODEL_URI = "models:/CustomerChurnModel/1"
DECISION_THRESHOLD = 0.32


def main():
    print("Loading registered model...")

    model = mlflow.sklearn.load_model(MODEL_URI)

    print("Registered model loaded successfully.")
    print("Model URI:", MODEL_URI)

    # Load the same feature data used during training
    df = load_features()

    X, y, _ = prepare_features(df)

    # Take one customer as a prediction example
    sample = X.iloc[[0]]

    probability = model.predict_proba(sample)[:, 1][0]
    prediction = int(probability >= DECISION_THRESHOLD)

    print("\n" + "=" * 50)
    print("REGISTERED MODEL PREDICTION")
    print("=" * 50)
    print(f"Churn probability : {probability:.4f}")
    print(f"Decision threshold: {DECISION_THRESHOLD:.2f}")
    print(f"Predicted churn   : {prediction}")

    print("\nPrediction meaning:")
    if prediction == 1:
        print("Customer is predicted to churn.")
    else:
        print("Customer is predicted to stay.")


if __name__ == "__main__":
    main()