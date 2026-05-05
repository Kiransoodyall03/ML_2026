import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score
from transformers import SequentialSensorImputer
from train import extract_features # Re-use the exact same logic!

import warnings
warnings.filterwarnings('ignore') # Clean terminal output

def load_and_prep_data():
    print("Loading and prepping data for Optuna...")
    df_data = pd.read_parquet("data/raw/train_data.parquet", engine='pyarrow')
    df_labels = pd.read_csv("data/raw/train_labels.csv")
    
    if 'Sample_ID' not in df_data.columns:
        df_data = df_data.reset_index()

    # Use INTERPOLATE to protect the FFT features
    imputer = SequentialSensorImputer(method='interpolate', group_col='Sample_ID')
    df_imputed = imputer.fit_transform(df_data)

    df_features = extract_features(df_imputed)
    
    label_sample_col = [c for c in df_labels.columns if 'sample' in c.lower()][0]
    target_col = [c for c in df_labels.columns if 'class' in c.lower()][0]
    
    df_final = pd.merge(df_features, df_labels.rename(columns={label_sample_col: 'Sample_ID'}), on='Sample_ID')
    
    X = df_final.drop(columns=['Sample_ID', target_col])
    y = df_final[target_col] - 1 # Shift for XGBoost
    
    return X, y

# Load data once into memory
X_global, y_global = load_and_prep_data()

def objective(trial):
    """Optuna will try to maximize the score returned by this function."""
    
    # 1. Let Optuna suggest hyperparameter values
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'mlogloss'
    }
    
    # 2. Build the model with the suggested params
    model = xgb.XGBClassifier(**param)
    
    # 3. Evaluate it using our exact competition metric
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, average='macro')
    
    scores = cross_val_score(model, X_global, y_global, cv=cv, scoring=scorer, n_jobs=-1)
    
    return np.mean(scores)

if __name__ == "__main__":
    print("\n=== Starting Optuna Hyperparameter Search ===")
    # Create a study object and tell it we want the HIGHEST score
    study = optuna.create_study(direction='maximize')
    
    # Run 20 experiments (you can increase n_trials if you have time!)
    study.optimize(objective, n_trials=20)
    
    print("\n=== Tuning Complete! ===")
    print(f"Best Macro-F1 Score: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")