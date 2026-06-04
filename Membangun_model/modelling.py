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

# Inisialisasi DagsHub
# Kita pastikan token diambil dari environment variable DAGSHUB_TOKEN
# yang sudah kita set di GitHub Secrets
dagshub.init(
    repo_owner='heriwibowo-dev', 
    repo_name='SMSML_HeriWibowo', 
    mlflow=True
)

# Load Data 
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
df = pd.read_csv(os.path.join(root_dir, 'heart.csv'))

# Preprocessing & Training
X = df.drop(columns=['target'])
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Heart_Disease_Prediction")

# Gunakan context manager untuk run MLflow
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 1. Log Model
    mlflow.sklearn.log_model(model, "model")
    
    # 2. Artefak: Confusion Matrix
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()
    
    # 3. Artefak: Feature Importance
    plt.figure(figsize=(8,6))
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances.nlargest(10).plot(kind='barh')
    plt.title("Top 10 Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()
    
    # Log Metrik
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    
    print(f"Training selesai. Akurasi: {accuracy:.4f}")
    print("Artefak berhasil di-log ke DagsHub.")
