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

# --- 1. OTENTIKASI OTOMATIS ---
# Jika kita berada di GitHub Actions, variabel DAGSHUB_TOKEN sudah ada.
# Library dagshub akan membaca variabel tersebut secara otomatis.
# Kita tidak perlu lagi menulis 'token=' di dalam init() untuk menghindari TypeError.
dagshub.init(repo_owner='heriwibowo-dev', repo_name='SMSML_HeriWibowo', mlflow=True)

# Jika masih ada kendala, kita paksa MLflow menggunakan kredensial dari environment
if "DAGSHUB_TOKEN" in os.environ:
    os.environ["MLFLOW_TRACKING_USERNAME"] = "heriwibowo-dev"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]

# --- 2. LOAD DATA ---
# Menggunakan path relatif agar script bisa berjalan di mana saja
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
df = pd.read_csv(os.path.join(root_dir, 'heart.csv'))

# --- 3. PREPROCESSING & TRAINING ---
X = df.drop(columns=['target'])
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- 4. TRAINING & LOGGING ---
mlflow.set_experiment("Heart_Disease_Prediction")

with mlflow.start_run():
    # Training
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    
    # Log model ke MLflow
    mlflow.sklearn.log_model(model, "model")
    
    # Log Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, model.predict(X_test)), annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()
    
    # Log Feature Importance
    plt.figure(figsize=(8,6))
    pd.Series(model.feature_importances_, index=X.columns).nlargest(10).plot(kind='barh')
    plt.title("Top 10 Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()
    
    # Log Accuracy
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    print(f"Training selesai. Akurasi: {accuracy:.4f}")
