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

# --- 1. OTENTIKASI & INISIALISASI ---
dagshub_token = os.environ.get("DAGSHUB_TOKEN")

if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN tidak ditemukan di environment variable!")

# Set variabel lingkungan untuk DagsHub
os.environ['DAGSHUB_USER_TOKEN'] = dagshub_token

dagshub.init(
    repo_owner='heriwibowo-dev', 
    repo_name='SMSML_HeriWibowo', 
    mlflow=True
)

# --- 2. LOAD DATA ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'namadataset_preprocessing', 'heart.csv')

if not os.path.exists(data_path):
    raise FileNotFoundError(f"File data tidak ditemukan di: {data_path}")

df = pd.read_csv(data_path)

# --- 3. PREPROCESSING ---
X = df.drop(columns=['target'])
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- 4. TRAINING & LOGGING ---
mlflow.set_experiment("Heart_Disease_Prediction")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, model.predict(X_test)), annot=True, fmt='d', cmap='Blues')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()
    
    print(f"Training sukses! Akurasi: {accuracy:.4f}")
