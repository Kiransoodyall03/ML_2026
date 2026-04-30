import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from transformers import SequentialSensorImputer
from evaluation import evaluate_model_cv 

def extract_features(df):
    """Safely aggregates time steps into summary statistics."""
    print(f"--- Debug: Data columns entering extraction: {df.columns.tolist()}")
    
    # Use explicit names based on your train_data.parquet
    # We'll try to find 'Sample_ID' or 'Sample ID'
    potential_sample_cols = [c for c in df.columns if 'sample' in c.lower()]
    if not potential_sample_cols:
        raise ValueError(f"Could not find a Sample ID column in: {df.columns.tolist()}")
    
    sample_col = potential_sample_cols[0]
    signal_cols = [c for c in df.columns if 'signal' in c.lower()]
    
    print(f"--- Grouping by: {sample_col}")
    features = df.groupby(sample_col)[signal_cols].agg(['mean', 'std'])
    
    # Flatten names
    features.columns = ['_'.join(col).strip() for col in features.columns.values]
    return features.reset_index().rename(columns={sample_col: 'Sample_ID'})

def main():
    print("\n=== COMS3007A: Training Baseline Model ===")
    
    data_path = "data/raw/train_data.parquet" 
    labels_path = "data/raw/train_labels.csv"
    
# 1. Load Data
    print(f"1. Loading files...")
    df_data = pd.read_parquet(data_path, engine='pyarrow')
    df_labels = pd.read_csv(labels_path)

    # NEW: If Sample_ID is the index, move it into the columns
    if 'Sample_ID' not in df_data.columns:
        print("--- Detected Sample_ID in index, resetting...")
        df_data = df_data.reset_index()
    # 2. Imputation
    print("2. Running imputation...")
    # We pass the first column name explicitly to avoid detection errors
    first_col = df_data.columns[0] 
    imputer = SequentialSensorImputer(method='ffill', group_col=first_col)
    
    # CRITICAL: Ensure result stays a DataFrame with columns
    df_imputed = imputer.fit_transform(df_data)
    if isinstance(df_imputed, np.ndarray):
        df_imputed = pd.DataFrame(df_imputed, columns=df_data.columns)

    # 3. Feature Engineering
    df_features = extract_features(df_imputed)
    
    # 4. Merge
    print("3. Merging labels...")
    label_sample_col = [c for c in df_labels.columns if 'sample' in c.lower()][0]
    target_col = [c for c in df_labels.columns if 'class' in c.lower()][0]
    
    df_final = pd.merge(
        df_features, 
        df_labels.rename(columns={label_sample_col: 'Sample_ID'}), 
        on='Sample_ID'
    )
    
    X = df_final.drop(columns=['Sample_ID', target_col])
    y = df_final[target_col]
    
    print(f"--- Final feature set: {X.shape[1]} features across {len(X)} samples.")
    
    # 5. Model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    print("4. Starting Cross-Validation...")
    evaluate_model_cv(model, X, y, n_splits=5)

if __name__ == "__main__":
    main()