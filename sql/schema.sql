-- ============================================
-- CUSTOMER CHURN ML PLATFORM
-- DATABASE SCHEMA
-- ============================================


-- ============================================
-- 1. CUSTOMERS
-- ============================================

CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    gender VARCHAR(20),
    senior_citizen BOOLEAN,
    partner BOOLEAN,
    dependents BOOLEAN,
    tenure_months INTEGER,
    churn BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ============================================
-- 2. CUSTOMER SERVICES
-- ============================================

CREATE TABLE IF NOT EXISTS customer_services (
    customer_id VARCHAR(50) PRIMARY KEY,

    phone_service BOOLEAN,
    multiple_lines VARCHAR(30),

    internet_service VARCHAR(50),
    online_security VARCHAR(30),
    online_backup VARCHAR(30),
    device_protection VARCHAR(30),
    tech_support VARCHAR(30),

    streaming_tv VARCHAR(30),
    streaming_movies VARCHAR(30),

    CONSTRAINT fk_services_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
        ON DELETE CASCADE
);


-- ============================================
-- 3. CUSTOMER BILLING
-- ============================================

CREATE TABLE IF NOT EXISTS customer_billing (
    customer_id VARCHAR(50) PRIMARY KEY,

    contract_type VARCHAR(50),
    paperless_billing BOOLEAN,
    payment_method VARCHAR(100),

    monthly_charges NUMERIC(10,2),
    total_charges NUMERIC(10,2),

    CONSTRAINT fk_billing_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
        ON DELETE CASCADE
);


-- ============================================
-- 4. MODEL VERSIONS
-- ============================================

CREATE TABLE IF NOT EXISTS model_versions (
    model_version_id SERIAL PRIMARY KEY,

    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    algorithm VARCHAR(100),

    accuracy NUMERIC(6,4),
    precision_score NUMERIC(6,4),
    recall_score NUMERIC(6,4),
    f1_score NUMERIC(6,4),
    roc_auc NUMERIC(6,4),

    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ============================================
-- 5. PREDICTIONS
-- ============================================

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id SERIAL PRIMARY KEY,

    customer_id VARCHAR(50) NOT NULL,

    model_version_id INTEGER,

    churn_probability NUMERIC(8,6),
    predicted_churn BOOLEAN,

    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_prediction_customer
        FOREIGN KEY (customer_id)
        REFERENCES customers(customer_id)
        ON DELETE CASCADE,

    CONSTRAINT fk_prediction_model
        FOREIGN KEY (model_version_id)
        REFERENCES model_versions(model_version_id)
);
