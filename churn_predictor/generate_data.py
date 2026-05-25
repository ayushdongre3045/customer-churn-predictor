"""
generate_data.py
Generates a realistic synthetic customer churn dataset and saves it to data/customers.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 2000

age = np.random.randint(18, 75, N)
tenure_months = np.random.randint(1, 72, N)
monthly_charges = np.round(np.random.uniform(20, 120, N), 2)
total_charges = np.round(monthly_charges * tenure_months + np.random.normal(0, 50, N), 2)
total_charges = np.clip(total_charges, 0, None)

num_products = np.random.choice([1, 2, 3, 4], N, p=[0.3, 0.35, 0.25, 0.1])
support_calls = np.random.poisson(2, N)

contract = np.random.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.25, 0.20])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], N,
    p=[0.35, 0.22, 0.22, 0.21]
)
internet_service = np.random.choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22])
gender = np.random.choice(["Male", "Female"], N)
senior_citizen = np.random.choice([0, 1], N, p=[0.84, 0.16])
has_dependents = np.random.choice(["Yes", "No"], N, p=[0.30, 0.70])
has_partner = np.random.choice(["Yes", "No"], N, p=[0.48, 0.52])

# Churn probability based on risk factors
churn_score = (
    0.3 * (contract == "Month-to-month").astype(float)
    - 0.2 * (contract == "Two year").astype(float)
    + 0.15 * (internet_service == "Fiber optic").astype(float)
    + 0.1 * (payment_method == "Electronic check").astype(float)
    + 0.008 * support_calls
    - 0.005 * tenure_months
    + 0.002 * monthly_charges
    - 0.05 * (num_products >= 3).astype(float)
    + 0.05 * senior_citizen
    + np.random.normal(0, 0.1, N)
)
churn_prob = 1 / (1 + np.exp(-churn_score * 3))
churn = (np.random.uniform(0, 1, N) < churn_prob).astype(int)

df = pd.DataFrame({
    "customer_id": [f"CUST{str(i).zfill(5)}" for i in range(1, N + 1)],
    "gender": gender,
    "senior_citizen": senior_citizen,
    "has_partner": has_partner,
    "has_dependents": has_dependents,
    "age": age,
    "tenure_months": tenure_months,
    "contract": contract,
    "payment_method": payment_method,
    "internet_service": internet_service,
    "num_products": num_products,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "support_calls": support_calls,
    "churn": churn,
})

# Inject a few missing values for realism
for col in ["total_charges", "monthly_charges", "support_calls"]:
    idx = np.random.choice(df.index, size=20, replace=False)
    df.loc[idx, col] = np.nan

Path("data").mkdir(exist_ok=True)
df.to_csv("data/customers.csv", index=False)
print(f"Dataset saved → data/customers.csv  ({N} rows, churn rate: {churn.mean():.1%})")
