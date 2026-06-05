import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dagshub

# 1. PENGATURAN OTORISASI (Wajib untuk GitHub Actions)
# Jika DAGSHUB_TOKEN tersedia di environment (dari Secrets), kita masukkan ke MLflow
if "DAGSHUB_TOKEN" in os.environ:
    os.environ["MLFLOW_TRACKING_USERNAME"] = "heriwibowo-dev" 
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
    # Opsional: Jika library membutuhkan ini secara eksplisit
    os.environ["DAGSHUB_USER_TOKEN"] = os.environ["DAGSHUB_TOKEN"]

# 2. Inisialisasi DagsHub
# Tanpa argumen 'token=' untuk menghindari TypeError
dagshub.init(repo_owner='heriwibowo-dev', repo_name='SMSML_HeriWibowo', mlflow=True)

# 3. Load Data
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
df = pd.read_csv(os.path.join(root_dir, 'heart.csv'))

# 4. Preprocessing & Training
X = df.drop(columns=['target'])
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Heart_Disease_Prediction")

# 5. Training & Logging
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    
    mlflow.sklearn.log_model(model, "model")
    
    # Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, model.predict(X_test)), annot=True, fmt='d', cmap='Blues')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()
    
    # Feature Importance
    plt.figure(figsize=(8,6))
    pd.Series(model.feature_importances_, index=X.columns).nlargest(10).plot(kind='barh')
    plt.title("Top 10 Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()
    
    # Log Metric
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    print(f"Training berhasil! Akurasi: {accuracy}")
