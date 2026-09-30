from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "user_activity.csv"
MODEL_FILE = ROOT / "models" / "isolation_forest.pkl"

st.set_page_config(page_title="User Behaviour Anomaly Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ User Behaviour Anomaly Detection")
st.caption("Machine Learning for Cybersecurity | Isolation Forest")

if not DATA_FILE.exists() or not MODEL_FILE.exists():
    st.warning("Model and dataset are not created yet.")
    st.code(r"venv\Scripts\python.exe src\train_model.py")
    st.stop()

data = pd.read_csv(DATA_FILE)
bundle = joblib.load(MODEL_FILE)
model = bundle["model"]
features = bundle["features"]

def risk_score(raw):
    return int(np.clip(50 - float(raw) * 100, 0, 100))

def risk_level(score):
    if score <= 30:
        return "Low"
    if score <= 60:
        return "Medium"
    if score <= 80:
        return "High"
    return "Critical"

def reasons(row):
    out = []
    if row["failed_login_count"] >= 4:
        out.append("Multiple failed login attempts")
    if row["files_accessed"] >= 150:
        out.append("Unusually high file access")
    if row["download_mb"] >= 1000:
        out.append("Unusually large download volume")
    if row["new_device"] == 1:
        out.append("New device detected")
    if row["new_ip"] == 1:
        out.append("New IP detected")
    if row["unique_ips"] >= 3:
        out.append("Multiple IP addresses observed")
    if row["login_hour"] <= 4 or row["login_hour"] >= 23:
        out.append("Login time differs from the learned pattern")
    if not out:
        out.append("Combined behaviour is unusual compared with the learned data")
    return out

data["status"] = data["prediction"].map({1: "Normal", -1: "Anomaly"})
data["risk_score"] = data["model_score"].apply(risk_score)
data["risk_level"] = data["risk_score"].apply(risk_level)

st.sidebar.header("🔎 Investigation")
selected_user = st.sidebar.selectbox("Select User ID", ["All"] + sorted(data["user_id"].unique()))
status_filter = st.sidebar.selectbox("Show records", ["All", "Anomaly", "Normal"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Events", len(data))
c2.metric("Normal Events", int((data.status == "Normal").sum()))
c3.metric("Anomalies", int((data.status == "Anomaly").sum()))
c4.metric("High/Critical Risk", int((data.risk_score >= 61).sum()))

st.divider()
st.subheader("🚨 Automatically Detected Behavioural Anomalies")

anomalies = data[data.status == "Anomaly"].copy()
cols = ["user_id", "timestamp", "login_hour", "failed_login_count", "files_accessed",
        "download_mb", "new_device", "new_ip", "risk_score", "risk_level"]
st.dataframe(anomalies[cols].sort_values("risk_score", ascending=False), use_container_width=True, hide_index=True)

st.divider()
st.subheader("👤 User Investigation")

if selected_user == "All":
    user_data = data
else:
    user_data = data[data.user_id == selected_user]

u1, u2, u3 = st.columns(3)
u1.metric("Activities", len(user_data))
u2.metric("Anomalies", int((user_data.status == "Anomaly").sum()))
u3.metric("Highest Risk", int(user_data.risk_score.max()))

if selected_user != "All" and len(user_data):
    last = user_data.sort_values("timestamp").iloc[-1]
    st.write(f"### User ID: {selected_user}")
    st.write(f"**Latest activity:** {last['timestamp']}")
    st.write(f"**Status:** {last['status']}")
    st.write(f"**Risk:** {last['risk_score']}/100 ({last['risk_level']})")
    st.write("**Behaviour details:**")
    st.write(
        f"- Login hour: {int(last['login_hour'])}:00\n"
        f"- Failed logins: {int(last['failed_login_count'])}\n"
        f"- Files accessed: {int(last['files_accessed'])}\n"
        f"- Download: {float(last['download_mb']):.1f} MB\n"
        f"- New device: {'Yes' if last['new_device'] else 'No'}\n"
        f"- New IP: {'Yes' if last['new_ip'] else 'No'}"
    )
    if last["status"] == "Anomaly":
        st.error("⚠️ Behavioural anomaly requires investigation.")
        for item in reasons(last):
            st.write("• " + item)
    else:
        st.success("Normal behavioural pattern.")

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("📊 Normal vs Anomaly")
    counts = data.status.value_counts().reset_index()
    counts.columns = ["status", "count"]
    st.plotly_chart(px.bar(counts, x="status", y="count"), use_container_width=True)

with right:
    st.subheader("📈 Files vs Downloads")
    st.plotly_chart(
        px.scatter(data, x="files_accessed", y="download_mb", color="status",
                   hover_data=["user_id", "timestamp", "failed_login_count"]),
        use_container_width=True
    )

st.divider()
st.subheader("🧪 Simulation Mode")
st.caption("Optional live-demo feature; the main workflow automatically analyzes the dataset.")

a, b, c, d = st.columns(4)
with a:
    sim_login = st.number_input("Login hour", 0, 23, 10)
    sim_failed = st.number_input("Failed logins", 0, 50, 0)
with b:
    sim_files = st.number_input("Files accessed", 0, 2000, 20)
    sim_download = st.number_input("Download MB", 0.0, 20000.0, 50.0)
with c:
    sim_device = st.selectbox("New device", ["No", "Yes"])
    sim_ip = st.selectbox("New IP", ["No", "Yes"])
with d:
    sim_duration = st.number_input("Session duration", 1, 1000, 45)
    sim_unique = st.number_input("Unique IPs", 1, 50, 1)

if st.button("Analyze New Activity"):
    x = pd.DataFrame([{
        "login_hour": sim_login,
        "failed_login_count": sim_failed,
        "files_accessed": sim_files,
        "download_mb": sim_download,
        "new_device": 1 if sim_device == "Yes" else 0,
        "new_ip": 1 if sim_ip == "Yes" else 0,
        "session_duration": sim_duration,
        "unique_ips": sim_unique
    }])
    pred = int(model.predict(x[features])[0])
    raw = float(model.decision_function(x[features])[0])
    score = risk_score(raw)
    level = risk_level(score)
    if pred == -1:
        st.error(f"🚨 BEHAVIOURAL ANOMALY | Risk: {score}/100 | {level}")
        for item in reasons(x.iloc[0]):
            st.write("• " + item)
    else:
        st.success(f"✅ NORMAL BEHAVIOUR | Risk: {score}/100 | {level}")

st.caption("Risk scores are prototype prioritization indicators, not proof of malicious intent.")
