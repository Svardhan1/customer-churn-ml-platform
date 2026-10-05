from src.data.load_features import load_features
from src.features.prepare import prepare_features


def main():
    # Load feature data from PostgreSQL
    df = load_features()

    # Prepare X, y and preprocessing pipeline
    X, y, preprocessor = prepare_features(df)

    print("\n--- ORIGINAL FEATURES ---")
    print("X shape:", X.shape)

    print("\n--- TARGET ---")
    print("y shape:", y.shape)
    print("Churn distribution:")
    print(y.value_counts())

    # Fit and transform the preprocessing pipeline
    X_processed = preprocessor.fit_transform(X)

    print("\n--- PROCESSED FEATURES ---")
    print("Processed shape:", X_processed.shape)

    print("\nPreprocessing successful.")


if __name__ == "__main__":
    main()