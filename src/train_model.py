from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
DATA.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

np.random.seed(42)
n = 1000
users = [f"U{1000+i}" for i in range(50)]

df = pd.DataFrame({
    "user_id": np.random.choice(users, n),
    "timestamp": pd.date_range("2026-09-01 08:00", periods=n, freq="30min"),
    "login_hour": np.random.choice(range(8, 19), n),
    "failed_login_count": np.random.poisson(0.4, n),
    "files_accessed": np.maximum(1, np.random.normal(25, 10, n)).round().astype(int),
    "download_mb": np.maximum(1, np.random.normal(80, 30, n)).round(1),
    "new_device": np.random.binomial(1, 0.04, n),
    "new_ip": np.random.binomial(1, 0.06, n),
    "session_duration": np.maximum(5, np.random.normal(45, 15, n)).round().astype(int),
    "unique_ips": np.random.choice([1, 1, 1, 2], n)
})

# Synthetic multi-feature anomalies for demonstration.
for i in range(40):
    df.loc[i, "login_hour"] = np.random.choice([0, 1, 2, 3, 23])
    df.loc[i, "failed_login_count"] = np.random.randint(5, 12)
    df.loc[i, "files_accessed"] = np.random.randint(200, 600)
    df.loc[i, "download_mb"] = np.random.randint(2000, 8000)
    df.loc[i, "new_device"] = 1
    df.loc[i, "new_ip"] = 1
    df.loc[i, "session_duration"] = np.random.randint(100, 300)
    df.loc[i, "unique_ips"] = np.random.randint(3, 7)

features = [
    "login_hour", "failed_login_count", "files_accessed", "download_mb",
    "new_device", "new_ip", "session_duration", "unique_ips"
]

model = IsolationForest(n_estimators=150, contamination=0.05, random_state=42)
model.fit(df[features])

df["prediction"] = model.predict(df[features])
df["model_score"] = model.decision_function(df[features])

df.to_csv(DATA / "user_activity.csv", index=False)
joblib.dump({"model": model, "features": features}, MODELS / "isolation_forest.pkl")

print("MODEL TRAINING COMPLETED")
print("Dataset:", DATA / "user_activity.csv")
print("Model:", MODELS / "isolation_forest.pkl")
print("Records:", len(df))
print("Anomalies:", int((df["prediction"] == -1).sum()))
