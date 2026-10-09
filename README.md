# Customer Churn Prediction ML Platform

A production-oriented machine learning project that predicts customer churn using a trained XGBoost model, PostgreSQL, MLflow, FastAPI, Docker, and AWS deployment infrastructure.

The goal is to demonstrate an end-to-end ML workflow, from data processing and model evaluation to API-based predictions, database persistence, containerization, and deployment.

## Project Overview

Customer churn occurs when customers stop using a company's products or services. Identifying customers who are likely to leave can help businesses prioritize retention strategies.

This project builds a machine learning platform that:

- Processes and validates customer data.
- Trains and evaluates churn prediction models.
- Tracks model experiments using MLflow.
- Serves predictions through a FastAPI REST API.
- Integrates PostgreSQL for customer, model-version, and prediction records.
- Packages the API and model in Docker.
- Uses AWS infrastructure for container deployment.

## Architecture

```text
Customer Churn Dataset
          |
          v
Data Cleaning and ETL
          |
          v
PostgreSQL Database
          |
          v
Feature Engineering
          |
          v
Model Training and Evaluation
          |
          v
MLflow Experiment Tracking
          |
          v
Trained XGBoost Model
          |
          v
FastAPI Prediction Service
          |
          v
Docker Container
          |
          v
Amazon ECR --> Amazon ECS
          |
          v
PostgreSQL Prediction Persistence
```

The architecture describes the intended workflow. External API connectivity and a complete cloud prediction-to-database round trip have not been fully verified.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn, XGBoost |
| Experiment tracking | MLflow |
| Database | PostgreSQL, SQLAlchemy |
| API | FastAPI, Uvicorn |
| Testing | Pytest |
| Containerization | Docker |
| Cloud | AWS ECR, ECS, RDS, Secrets Manager, CloudWatch |
| Version control | Git, GitHub |

## Dataset

The project uses the Telco Customer Churn dataset.

- **Records:** 7,043 customers
- **Original columns:** 21
- **Target:** `Churn`

The dataset contains customer demographics, service subscriptions, account information, billing details, and churn labels.

The preprocessing workflow handles blank `TotalCharges` values and prepares numerical and categorical features for modelling.

The raw dataset is not included in the repository. Obtain it from the original dataset source and place it at the path expected by the project.

## Machine Learning Workflow

1. Load and inspect the dataset.
2. Clean and validate customer records.
3. Prepare numerical and categorical features.
4. Split the data into training and test sets.
5. Train and compare candidate classification models.
6. Tune model parameters and analyze prediction thresholds.
7. Track experiments and model artifacts with MLflow.
8. Evaluate the selected model on the held-out test set.
9. Load the trained model into the prediction API.

The final XGBoost model uses a decision threshold selected through out-of-fold validation rather than relying automatically on the default 0.50 threshold.

## Model Evaluation

The reported held-out test results at a decision threshold of 0.32 are:

| Metric | Result |
|---|---:|
| Accuracy | 75.87% |
| Churn precision | 53.31% |
| Churn recall | 73.26% |
| Churn F1-score | 61.71% |
| ROC-AUC | 84.17% |
| PR-AUC | 65.91% |

These results reflect the current evaluation run and may vary if the data split, dependencies, or training configuration changes.

Recall is important in churn prediction because missed churners can represent lost retention opportunities. Precision is also important because incorrectly flagging customers can waste retention resources.

## Database Design

PostgreSQL stores structured project data using tables for:

- `customers`
- `customer_services`
- `customer_billing`
- `model_versions`
- `predictions`

The database layer supports database connectivity, model-version registration, and prediction persistence.

Database credentials should be provided through environment variables or a secrets manager. Never commit real credentials to GitHub.

## REST API

The FastAPI service provides prediction functionality.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check |
| `POST /predict` | Generate a churn prediction |

Interactive API documentation is normally available at `/docs` when the API is running and reachable.

Example local health-check request:

```bash
curl http://localhost:8000/health
```

A prediction request must follow the input schema defined in `src/api/main.py`. Refer to that file for the required fields and validation rules.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Svardhan1/customer-churn-ml-platform.git
cd customer-churn-ml-platform
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file using `.env.example` as a template.

Configure the PostgreSQL connection values required by the application:

```dotenv
DB_HOST=localhost
DB_PORT=5432
DB_NAME=customer_churn_db
DB_USER=postgres
DB_PASSWORD=YOUR_LOCAL_PASSWORD
```

Use your own local database credentials. Do not commit `.env`.

### 5. Prepare the database

Create the database and apply the SQL schema in `sql/schema.sql`. Configure the remaining data-loading steps according to the scripts in `src/data/`.

### 6. Run the API

Use the project's FastAPI entry point:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

This command assumes the model artifact is available at the path expected by the application and the required environment variables are configured.

### 7. Run tests

```bash
python -m pytest
```

## Docker

Build the API image from the project root using the project's Dockerfile:
### Model Artifacts

The Docker image packages the trained MLflow model artifacts from the local `mlruns/` directory. These generated artifacts are excluded from Git to keep the repository lightweight.

Before building the image, ensure the expected model artifacts are available locally. They can be generated through the project's training workflow or supplied from a compatible MLflow model export.

```bash
docker build -f docker/Dockerfile -t customer-churn-api:latest .
```

Run the container with the required environment variables and network access to PostgreSQL configured for your environment.

Do not embed database passwords or cloud credentials in the image.

## AWS Deployment

The project includes deployment work using:

- **Amazon ECR** to store the container image.
- **Amazon ECS** to run the containerized API.
- **Amazon RDS for PostgreSQL** for persistent storage.
- **AWS Secrets Manager** for database credentials.
- **Amazon CloudWatch** for application logs.

The ECS container successfully started and loaded the model, and application startup was observed in CloudWatch logs. However, external API connectivity and end-to-end prediction persistence from the deployed service were not fully verified at the time of writing.

The deployment should therefore be considered partially verified rather than a confirmed, fully operational public API.

## Repository Structure

```text
customer-churn-ml-platform/
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── docker/
├── notebooks/
├── scripts/
├── sql/
├── src/
│   ├── api/
│   ├── data/
│   ├── db/
│   ├── features/
│   ├── models/
│   └── monitoring/
├── tests/
├── .dockerignore
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

This is a representative structure; actual contents may vary as the project evolves.

## Current Limitations and Future Improvements

- Complete external connectivity verification for the deployed API.
- Verify end-to-end prediction persistence in the cloud PostgreSQL database.
- Add automated CI testing and linting.
- Add model and data drift monitoring.
- Improve API integration and load testing.
- Add reproducible model-training and deployment workflows.

## Learning Outcomes

This project brings together practical skills in:

- Data cleaning and ETL
- Machine learning and classification evaluation
- Threshold tuning and imbalanced classification
- MLflow experiment tracking
- REST API development
- PostgreSQL integration
- Docker image creation
- AWS container deployment and logging
- Automated testing and Git-based development

## Disclaimer

This is a portfolio and learning project. Churn predictions are probabilistic estimates and should be evaluated against business requirements before being used for operational decisions.