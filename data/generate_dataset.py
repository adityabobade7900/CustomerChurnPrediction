"""
Generates a realistic IBM Telco-style Customer Churn dataset (7043 records).
Run once: python generate_dataset.py
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 7043

# Customer IDs
customer_ids = [f"CUST-{str(i).zfill(5)}" for i in range(1, n + 1)]

# Demographics
gender = np.random.choice(["Male", "Female"], n)
senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], n)
dependents = np.random.choice(["Yes", "No"], n, p=[0.30, 0.70])

# Tenure (months) — bimodal: new customers + long-term
part1 = np.random.randint(1, 12, int(n * 0.30))
part2 = np.random.randint(12, 36, int(n * 0.25))
part3 = np.random.randint(36, 72, n - len(part1) - len(part2))
tenure = np.concatenate([part1, part2, part3])
np.random.shuffle(tenure)

# Services
phone_service = np.random.choice(["Yes", "No"], n, p=[0.90, 0.10])
multiple_lines = np.where(phone_service == "No", "No phone service",
                          np.random.choice(["Yes", "No"], n))
internet_service = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
online_security = np.where(internet_service == "No", "No internet service",
                            np.random.choice(["Yes", "No"], n, p=[0.29, 0.71]))
online_backup = np.where(internet_service == "No", "No internet service",
                          np.random.choice(["Yes", "No"], n, p=[0.34, 0.66]))
device_protection = np.where(internet_service == "No", "No internet service",
                              np.random.choice(["Yes", "No"], n, p=[0.34, 0.66]))
tech_support = np.where(internet_service == "No", "No internet service",
                         np.random.choice(["Yes", "No"], n, p=[0.29, 0.71]))
streaming_tv = np.where(internet_service == "No", "No internet service",
                         np.random.choice(["Yes", "No"], n, p=[0.38, 0.62]))
streaming_movies = np.where(internet_service == "No", "No internet service",
                              np.random.choice(["Yes", "No"], n, p=[0.39, 0.61]))

# Contract & billing
contract = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
paperless_billing = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    n, p=[0.34, 0.23, 0.22, 0.21]
)

# Monthly charges based on services
base_charge = np.where(internet_service == "No", 20,
               np.where(internet_service == "DSL", 45, 70))
addon_charge = (
    (multiple_lines == "Yes").astype(int) * 10 +
    (online_security == "Yes").astype(int) * 6 +
    (online_backup == "Yes").astype(int) * 6 +
    (device_protection == "Yes").astype(int) * 6 +
    (tech_support == "Yes").astype(int) * 6 +
    (streaming_tv == "Yes").astype(int) * 8 +
    (streaming_movies == "Yes").astype(int) * 8
)
monthly_charges = (base_charge + addon_charge + np.random.normal(0, 3, n)).round(2)
monthly_charges = np.clip(monthly_charges, 18.25, 118.75)
total_charges = (monthly_charges * tenure + np.random.normal(0, 10, n)).round(2)
total_charges = np.clip(total_charges, 18.80, 8684.80)

# Churn — realistic probabilities
churn_prob = np.zeros(n)
churn_prob += np.where(contract == "Month-to-month", 0.25, 0.0)
churn_prob += np.where(internet_service == "Fiber optic", 0.10, 0.0)
churn_prob += np.where(tenure < 12, 0.15, np.where(tenure > 48, -0.10, 0.0))
churn_prob += np.where(payment_method == "Electronic check", 0.08, 0.0)
churn_prob += np.where(online_security == "No", 0.05, -0.03)
churn_prob += np.where(tech_support == "No", 0.05, -0.03)
churn_prob += np.where(monthly_charges > 70, 0.05, 0.0)
churn_prob = np.clip(churn_prob, 0.03, 0.75)
churn = np.array(["Yes" if np.random.random() < p else "No" for p in churn_prob])

df = pd.DataFrame({
    "customerID": customer_ids,
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churn
})

df.to_csv("telco_churn.csv", index=False)
print(f"Dataset created: {len(df)} records")
print(f"Churn rate: {(df['Churn']=='Yes').mean():.2%}")
