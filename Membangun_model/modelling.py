import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# 1. Load Data
# Pastikan file heart.csv sudah ada di root folder repositori Anda
df = pd.read_csv('heart.csv')

# 2. Preprocessing
X = df.drop(columns=['target'])
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Training & Tracking dengan MLFlow
# MLFlow akan menggunakan kredensial dari GitHub Secrets yang sudah Anda set
mlflow.set_experiment("Heart_Disease_Prediction")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Log model ke MLFlow
    mlflow.sklearn.log_model(model, "model")
    
    # Log metrik (contoh akurasi)
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    print(f"Model berhasil dilatih dengan akurasi: {accuracy}")
