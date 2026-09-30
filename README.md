# User Behaviour Anomaly Detection

Simple academic cybersecurity ML prototype.

The system analyzes user activity records, identifies anomalous behaviour with Isolation Forest, and shows the User ID, activity details, risk and reasons in a dashboard.

The dataset is synthetic for hackathon demonstration. A 2 AM login alone is not automatically malicious; the model considers multiple features together.

## Run in VS Code

Open this folder in VS Code and open Terminal.

```powershell
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe src\train_model.py
venv\Scripts\python.exe -m streamlit run app.py
```

If PowerShell blocks activation, you do not need to activate the venv. The commands above directly use its Python.

The dashboard normally opens at http://localhost:8501.

## Main workflow

Synthetic activity records
-> preprocessing
-> feature engineering
-> Isolation Forest
-> anomaly detection
-> User ID + unusual activity
-> risk prioritization
-> dashboard

The manual Simulation Mode is only an optional live-demo feature.
