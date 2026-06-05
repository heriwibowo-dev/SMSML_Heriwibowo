import os
# SET VARIABEL LINGKUNGAN PALING ATAS SEBELUM IMPORT DAGSHUB
os.environ['DAGSHUB_USER_TOKEN'] = os.environ.get("DAGSHUB_TOKEN", "")

import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Inisialisasi dengan force_init=True
dagshub.init(
    repo_owner='heriwibowo-dev', 
    repo_name='SMSML_HeriWibowo', 
    mlflow=True,
    force_init=True
)

# --- LOAD & PREPROCESS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'namadataset_preprocessing', 'heart.csv')
df = pd.read_csv(data_path)

X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.2, random_state=42)

# --- TUNING ---
mlflow.set_experiment("Heart_Disease_Tuning")
with mlflow.start_run():
    grid = GridSearchCV(RandomForestClassifier(random_state=42), 
                        {'n_estimators': [50, 100], 'max_depth': [None, 10]}, cv=3)
    grid.fit(X_train, y_train)
    
    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("best_accuracy", grid.best_score_)
    mlflow.sklearn.log_model(grid.best_estimator_, "best_model")
    print(f"Tuning berhasil dengan akurasi: {grid.best_score_:.4f}")
