import mlflow
from mlflow import MlflowClient

EXPERIMENT_ID = "1"
RUN_ID = "6e02b59cb2434cec90094d54ec531bc4"
MODEL_NAME = "CustomerChurnModel"


def main():
    client = MlflowClient()

    model_uri = f"runs:/{RUN_ID}/churn_pipeline"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    print("\n" + "=" * 50)
    print("MODEL REGISTERED SUCCESSFULLY")
    print("=" * 50)
    print("Model name:", MODEL_NAME)
    print("Version:", registered_model.version)
    print("Run ID:", RUN_ID)
    print("Model URI:", model_uri)


if __name__ == "__main__":
    main()