-- ============================================================
-- Customer Churn Prediction — Business Queries
-- Author : Aditya Bobade
-- Database: churn_db
-- ============================================================

USE churn_db;

-- ─────────────────────────────────────────────────────────────
-- Q1: Overall churn rate
-- ─────────────────────────────────────────────────────────────
SELECT
    COUNT(*)                                        AS total_customers,
    SUM(churn = 'Yes')                             AS churned,
    SUM(churn = 'No')                              AS retained,
    ROUND(AVG(churn = 'Yes') * 100, 2)            AS churn_rate_pct
FROM customers;

-- ─────────────────────────────────────────────────────────────
-- Q2: Churn rate by contract type
-- ─────────────────────────────────────────────────────────────
SELECT
    contract,
    COUNT(*)                                AS total,
    SUM(churn = 'Yes')                     AS churned,
    ROUND(AVG(churn = 'Yes') * 100, 2)    AS churn_rate_pct
FROM customers
GROUP BY contract
ORDER BY churn_rate_pct DESC;

-- ─────────────────────────────────────────────────────────────
-- Q3: Churn rate by internet service type
-- ─────────────────────────────────────────────────────────────
SELECT
    internet_service,
    COUNT(*)                                AS total,
    SUM(churn = 'Yes')                     AS churned,
    ROUND(AVG(churn = 'Yes') * 100, 2)    AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)         AS avg_monthly_charge
FROM customers
GROUP BY internet_service
ORDER BY churn_rate_pct DESC;

-- ─────────────────────────────────────────────────────────────
-- Q4: Average tenure and charges by churn status
-- ─────────────────────────────────────────────────────────────
SELECT
    churn,
    ROUND(AVG(tenure), 1)                  AS avg_tenure_months,
    ROUND(AVG(monthly_charges), 2)         AS avg_monthly_charge,
    ROUND(AVG(total_charges), 2)           AS avg_total_charge,
    COUNT(*)                               AS customer_count
FROM customers
GROUP BY churn;

-- ─────────────────────────────────────────────────────────────
-- Q5: High-risk customers with monthly revenue exposure
-- ─────────────────────────────────────────────────────────────
SELECT
    c.customer_id,
    c.contract,
    c.internet_service,
    c.tenure,
    c.monthly_charges,
    r.risk_score,
    r.risk_tier
FROM customers c
JOIN customer_risk_scores r ON c.customer_id = r.customer_id
WHERE r.risk_tier IN ('High Risk', 'Critical Risk')
ORDER BY r.risk_score DESC, c.monthly_charges DESC
LIMIT 50;

-- ─────────────────────────────────────────────────────────────
-- Q6: Monthly revenue at risk by tier
-- ─────────────────────────────────────────────────────────────
SELECT
    r.risk_tier,
    COUNT(*)                               AS customers,
    ROUND(SUM(c.monthly_charges), 2)       AS total_monthly_revenue,
    ROUND(AVG(c.monthly_charges), 2)       AS avg_monthly_charge,
    ROUND(AVG(r.risk_score), 1)            AS avg_risk_score
FROM customers c
JOIN customer_risk_scores r ON c.customer_id = r.customer_id
GROUP BY r.risk_tier
ORDER BY FIELD(r.risk_tier, 'Low Risk','Medium Risk','High Risk','Critical Risk');

-- ─────────────────────────────────────────────────────────────
-- Q7: Senior citizens churn analysis
-- ─────────────────────────────────────────────────────────────
SELECT
    senior_citizen,
    COUNT(*)                                AS total,
    SUM(churn = 'Yes')                     AS churned,
    ROUND(AVG(churn = 'Yes') * 100, 2)    AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)         AS avg_monthly_charge
FROM customers
GROUP BY senior_citizen;

-- ─────────────────────────────────────────────────────────────
-- Q8: Churn rate by payment method
-- ─────────────────────────────────────────────────────────────
SELECT
    payment_method,
    COUNT(*)                                AS total,
    SUM(churn = 'Yes')                     AS churned,
    ROUND(AVG(churn = 'Yes') * 100, 2)    AS churn_rate_pct
FROM customers
GROUP BY payment_method
ORDER BY churn_rate_pct DESC;

-- ─────────────────────────────────────────────────────────────
-- Q9: Tenure cohort churn analysis (window function)
-- ─────────────────────────────────────────────────────────────
SELECT
    tenure_group,
    COUNT(*)                                AS total,
    SUM(is_churned)                        AS churned,
    ROUND(AVG(is_churned) * 100, 2)        AS churn_rate_pct,
    ROUND(AVG(monthly_charges), 2)         AS avg_monthly_charge
FROM (
    SELECT *,
           (churn = 'Yes')                 AS is_churned,
           CASE
               WHEN tenure BETWEEN 0  AND 12 THEN '0–12 months'
               WHEN tenure BETWEEN 13 AND 24 THEN '13–24 months'
               WHEN tenure BETWEEN 25 AND 48 THEN '25–48 months'
               ELSE '49+ months'
           END                             AS tenure_group
    FROM customers
) AS cohorts
GROUP BY tenure_group
ORDER BY MIN(tenure);

-- ─────────────────────────────────────────────────────────────
-- Q10: High-value customers (top 25% charges) at critical risk
-- ─────────────────────────────────────────────────────────────
WITH ranked AS (
    SELECT *,
           NTILE(4) OVER (ORDER BY monthly_charges) AS quartile
    FROM customers
)

SELECT
    c.customer_id,
    c.contract,
    c.tenure,
    c.monthly_charges,
    c.internet_service,
    r.risk_score,
    r.risk_tier
FROM ranked c
JOIN customer_risk_scores r
    ON c.customer_id = r.customer_id
WHERE r.risk_tier = 'Critical Risk'
  AND c.quartile = 4
ORDER BY c.monthly_charges DESC;

-- ─────────────────────────────────────────────────────────────
-- Q11: Retention action success rate
-- ─────────────────────────────────────────────────────────────
SELECT
    action_type,
    COUNT(*)                                        AS total_actions,
    SUM(outcome = 'Retained')                      AS retained,
    SUM(outcome = 'Churned')                       AS churned,
    SUM(outcome = 'Pending')                       AS pending,
    ROUND(SUM(outcome='Retained') /
          NULLIF(SUM(outcome IN ('Retained','Churned')), 0) * 100, 2) AS retention_success_pct
FROM retention_actions
GROUP BY action_type
ORDER BY retention_success_pct DESC;

-- ─────────────────────────────────────────────────────────────
-- Q12: Month-over-month model performance tracking
-- ─────────────────────────────────────────────────────────────
SELECT
    model_name,
    model_version,
    precision_score,
    recall_score,
    f1_score,
    roc_auc,
    training_samples,
    DATE(trained_at)  AS trained_date
FROM model_performance
ORDER BY model_name, trained_at DESC;
