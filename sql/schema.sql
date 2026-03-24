-- Active: 1772884175900@@127.0.0.1@3306@churn_db
-- ============================================================
-- Customer Churn Prediction — MySQL Database Schema
-- Author : Aditya Bobade
-- Project: Customer Churn Prediction | IBM Telco Dataset
-- ============================================================

CREATE DATABASE IF NOT EXISTS churn_db;
USE churn_db;

-- ─────────────────────────────────────────────────────────────
-- Table 1: customers  (raw customer data)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS customers (
    customer_id         VARCHAR(12)    PRIMARY KEY,
    gender              ENUM('Male','Female')          NOT NULL,
    senior_citizen      TINYINT(1)                     NOT NULL DEFAULT 0,
    partner             ENUM('Yes','No')               NOT NULL,
    dependents          ENUM('Yes','No')               NOT NULL,
    tenure              INT                            NOT NULL,
    phone_service       ENUM('Yes','No')               NOT NULL,
    multiple_lines      VARCHAR(20)                    NOT NULL,
    internet_service    ENUM('DSL','Fiber optic','No') NOT NULL,
    online_security     VARCHAR(25)                    NOT NULL,
    online_backup       VARCHAR(25)                    NOT NULL,
    device_protection   VARCHAR(25)                    NOT NULL,
    tech_support        VARCHAR(25)                    NOT NULL,
    streaming_tv        VARCHAR(25)                    NOT NULL,
    streaming_movies    VARCHAR(25)                    NOT NULL,
    contract            ENUM('Month-to-month','One year','Two year') NOT NULL,
    paperless_billing   ENUM('Yes','No')               NOT NULL,
    payment_method      VARCHAR(40)                    NOT NULL,
    monthly_charges     DECIMAL(8,2)                   NOT NULL,
    total_charges       DECIMAL(10,2)                  NOT NULL,
    churn               ENUM('Yes','No')               NOT NULL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ─────────────────────────────────────────────────────────────
-- Table 2: customer_risk_scores  (ML output)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS customer_risk_scores (
    score_id            INT AUTO_INCREMENT PRIMARY KEY,
    customer_id         VARCHAR(12)    NOT NULL,
    churn_probability   DECIMAL(6,4)   NOT NULL,
    risk_score          DECIMAL(5,1)   NOT NULL,
    risk_tier           ENUM('Low Risk','Medium Risk','High Risk','Critical Risk') NOT NULL,
    model_version       VARCHAR(20)    NOT NULL DEFAULT 'v1.0',
    scored_at           TIMESTAMP      DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- ─────────────────────────────────────────────────────────────
-- Table 3: retention_actions  (business recommendations log)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS retention_actions (
    action_id           INT AUTO_INCREMENT PRIMARY KEY,
    customer_id         VARCHAR(12)    NOT NULL,
    action_type         ENUM('Discount Offer','Contract Upgrade',
                             'Proactive Outreach','Escalation',
                             'Loyalty Reward')         NOT NULL,
    action_date         DATE                           NOT NULL,
    agent_id            VARCHAR(10),
    outcome             ENUM('Retained','Churned','Pending') DEFAULT 'Pending',
    notes               TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- ─────────────────────────────────────────────────────────────
-- Table 4: model_performance  (track model versions over time)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_performance (
    perf_id             INT AUTO_INCREMENT PRIMARY KEY,
    model_name          VARCHAR(50)    NOT NULL,
    model_version       VARCHAR(20)    NOT NULL,
    precision_score     DECIMAL(6,4),
    recall_score        DECIMAL(6,4),
    f1_score            DECIMAL(6,4),
    roc_auc             DECIMAL(6,4),
    training_samples    INT,
    test_samples        INT,
    trained_at          TIMESTAMP      DEFAULT CURRENT_TIMESTAMP
);

-- ─────────────────────────────────────────────────────────────
-- Indexes for query performance
-- ─────────────────────────────────────────────────────────────
CREATE INDEX idx_churn         ON customers(churn);
CREATE INDEX idx_contract      ON customers(contract);
CREATE INDEX idx_internet      ON customers(internet_service);
CREATE INDEX idx_risk_tier     ON customer_risk_scores(risk_tier);
CREATE INDEX idx_risk_score    ON customer_risk_scores(risk_score);
CREATE INDEX idx_action_date   ON retention_actions(action_date);

-- ─────────────────────────────────────────────────────────────
-- Seed model_performance with current results
-- ─────────────────────────────────────────────────────────────
INSERT INTO model_performance
    (model_name, model_version, precision_score, recall_score, f1_score, roc_auc, training_samples, test_samples)
VALUES
    ('Logistic Regression', 'v1.0', 0.7030, 0.7050, 0.7040, 0.7600, 8017, 2005),
    ('Random Forest',       'v1.0', 0.8100, 0.7940, 0.8020, 0.8720, 8017, 2005),
    ('Gradient Boosting',   'v1.0', 0.7520, 0.7390, 0.7450, 0.8030, 8017, 2005);
