import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
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

# --- 4. HYPERPARAMETER TUNING ---
mlflow.set_experiment("Heart_Disease_Tuning")

with mlflow.start_run():
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_accuracy", grid_search.best_score_)
    mlflow.sklearn.log_model(best_model, "best_model")
    
    print(f"Tuning Selesai! Best Accuracy: {grid_search.best_score_:.4f}")
