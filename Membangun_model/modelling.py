import os
# Set variabel lingkungan agar dibaca otomatis oleh dagshub
os.environ['DAGSHUB_USER_TOKEN'] = os.environ.get("DAGSHUB_TOKEN", "")

import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Inisialisasi tanpa parameter yang berpotensi error
dagshub.init(
    repo_owner='heriwibowo-dev', 
    repo_name='SMSML_HeriWibowo', 
    mlflow=True
)

# --- LOAD & PREPROCESS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'namadataset_preprocessing', 'heart.csv')
df = pd.read_csv(data_path)

X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.2, random_state=42)

# --- TRAINING ---
mlflow.set_experiment("Heart_Disease_Prediction")
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    mlflow.log_metric("accuracy", model.score(X_test, y_test))
    mlflow.sklearn.log_model(model, "model")
    print("Training berhasil!")
