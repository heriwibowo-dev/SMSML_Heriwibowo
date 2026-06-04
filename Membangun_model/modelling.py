import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import dagshub # Tambahkan ini agar terhubung ke DagsHub

# 1. Inisialisasi Koneksi DagsHub
# Ini akan otomatis mengambil kredensial dari environment variable (Secrets)
dagshub.init(repo_owner='heriwibowo-dev', repo_name='SMSML_HeriWibowo', mlflow=True)

# 2. Load Data
script_dir = os.path.dirname(os.path.abspath(__file__))
# Jika script ada di folder Membangun_model, naik 1 level ke root
root_dir = os.path.dirname(script_dir)
data_path = os.path.join(root_dir, 'heart.csv')

df = pd.read_csv(data_path)

# 3. Preprocessing
X = df.drop(columns=['target'])
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Training & Tracking dengan MLFlow
mlflow.set_experiment("Heart_Disease_Prediction")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Log model ke MLFlow
    mlflow.sklearn.log_model(model, "model")
    
    # Log metrik
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    print(f"Model berhasil dilatih dengan akurasi: {accuracy}")
