import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

# 1. Inisialisasi DagsHub
# Library dagshub secara otomatis akan mengambil nilai dari environment variable
# DAGSHUB_TOKEN, MLFLOW_TRACKING_USERNAME, dan MLFLOW_TRACKING_PASSWORD
dagshub.init(repo_owner='heriwibowo-dev', repo_name='SMSML_HeriWibowo', mlflow=True)

# 2. Path dinamis (Lebih aman untuk CI/CD)
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
data_path = os.path.join(root_dir, 'heart.csv')

# Cek apakah file ada sebelum load
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File data tidak ditemukan di: {data_path}. Pastikan heart.csv berada di root direktori.")

df = pd.read_csv(data_path)

# 3. Preprocessing
X = df.drop(columns=['target'])
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Training & Logging
mlflow.set_experiment("Heart_Disease_Prediction")

with mlflow.start_run():
    # Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Logging
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metric("accuracy", model.score(X_test, y_test))
    
    # Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, model.predict(X_test)), annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    
    # Log artifact
    mlflow.log_artifact("confusion_matrix.png")
    
    plt.close()
    print("Training dan Logging ke DagsHub Berhasil!")
