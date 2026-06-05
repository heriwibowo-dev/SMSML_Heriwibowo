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

# Inisialisasi DagsHub tanpa argumen 'token'
# Library akan otomatis mengambil DAGSHUB_TOKEN dari environment variable
dagshub.init(repo_owner='heriwibowo-dev', repo_name='SMSML_HeriWibowo', mlflow=True)

# Load Data
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
df = pd.read_csv(os.path.join(root_dir, 'heart.csv'))

# Preprocessing & Training
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.2, random_state=42)

mlflow.set_experiment("Heart_Disease_Prediction")

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
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()
    
    mlflow.log_metric("accuracy", model.score(X_test, y_test))
