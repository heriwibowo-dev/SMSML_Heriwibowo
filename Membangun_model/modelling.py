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

# --- OTENTIKASI OTOMATIS ---
# dagshub.init() akan membaca DAGSHUB_TOKEN dari environment variable.
# Tidak ada argumen 'token' untuk menghindari TypeError.
dagshub.init(repo_owner='heriwibowo-dev', repo_name='SMSML_HeriWibowo', mlflow=True)

# --- TRAINING & LOGGING ---
# Load Data
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
df = pd.read_csv(os.path.join(root_dir, 'heart.csv'))

# Preprocessing
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    StandardScaler().fit_transform(X), y, test_size=0.2, random_state=42
)

mlflow.set_experiment("Heart_Disease_Prediction")

with mlflow.start_run():
    # Training
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    
    # Logging
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_metric("accuracy", model.score(X_test, y_test))
    
    # Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, model.predict(X_test)), annot=True, fmt='d', cmap='Blues')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()
    
    print("Training dan Logging ke DagsHub Berhasil!")
