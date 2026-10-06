import mlflow


def main():

    # Select our experiment
    mlflow.set_experiment(
        "Customer Churn Prediction"
    )

    # Start one MLflow run
    with mlflow.start_run():

        # Log a parameter
        mlflow.log_param(
            "model",
            "XGBoost"
        )

        # Log a metric
        mlflow.log_metric(
            "accuracy",
            0.7587
        )

        print("MLflow test run completed successfully.")


if __name__ == "__main__":
    main()