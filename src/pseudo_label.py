import pandas as pd
import numpy as np
import xgboost as xgb
import os
from transformers import SequentialSensorImputer
from train import extract_features # Import your existing feature extractor!

def main():
    print("=== Phase 1: Kaggle Pseudo-Labeling ===")
    
    # 1. Load Everything
    df_train = pd.read_parquet("data/raw/train_data.parquet", engine='pyarrow')
    df_labels = pd.read_csv("data/raw/train_labels.csv")
    df_test = pd.read_parquet("data/raw/test_data.parquet", engine='pyarrow') # The hidden data!
    
    if 'Sample_ID' not in df_train.columns: df_train = df_train.reset_index()
    if 'Sample_ID' not in df_test.columns: df_test = df_test.reset_index()

    # 2. Impute Both
    print("Imputing Train and Test sets...")
    imputer = SequentialSensorImputer(method='interpolate', group_col='Sample_ID')
    df_train_imp = imputer.fit_transform(df_train)
    df_test_imp = imputer.fit_transform(df_test)

    # 3. Extract Features
    print("Extracting FFT features...")
    X_train_feat = extract_features(df_train_imp)
    X_test_feat = extract_features(df_test_imp)

    # Merge labels for training
    target_col = [c for c in df_labels.columns if 'class' in c.lower()][0]
    train_full = pd.merge(X_train_feat, df_labels, on='Sample_ID')
    
    X_train = train_full.drop(columns=['Sample_ID', target_col])
    y_train = train_full[target_col] - 1 # XGBoost needs 0-5
    X_test = X_test_feat.drop(columns=['Sample_ID'])

    # 4. Train the Optimized XGBoost Model
    print("Training Master XGBoost model on Train Data...")
    model = xgb.XGBClassifier(
        n_estimators=353, max_depth=6, learning_rate=0.1472,
        subsample=0.9552, colsample_bytree=0.6650,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 5. Predict Probabilities on the Hidden Test Set
    print("Predicting on Hidden Test Data...")
    probs = model.predict_proba(X_test)
    
    # 6. Extract High-Confidence Pseudo-Labels (> 95% confident)
    confidence_threshold = 0.95
    max_probs = np.max(probs, axis=1)
    confident_indices = np.where(max_probs > confidence_threshold)[0]
    
    print(f"Found {len(confident_indices)} test samples with >95% confidence!")
    
    # Grab the confident samples and assign the predicted class
    pseudo_X = X_test.iloc[confident_indices].copy()
    pseudo_y = np.argmax(probs[confident_indices], axis=1)
    
    # Re-attach Sample IDs for saving
    pseudo_samples = X_test_feat.iloc[confident_indices]['Sample_ID'].values
    pseudo_labels_df = pd.DataFrame({'Sample_ID': pseudo_samples, target_col: pseudo_y + 1}) # Shift back to 1-6
    
    # 7. Combine and Save!
    print("Combining Train and Pseudo-Test Data...")
    combined_labels = pd.concat([df_labels, pseudo_labels_df], ignore_index=True)
    
    # Save the expanded dataset
    os.makedirs("data/processed", exist_ok=True)
    df_train_imp.to_parquet("data/processed/train_imputed.parquet")
    df_test_imp[df_test_imp['Sample_ID'].isin(pseudo_samples)].to_parquet("data/processed/pseudo_imputed.parquet")
    combined_labels.to_csv("data/processed/combined_labels.csv", index=False)
    
    print("Phase 1 Complete! Pseudo-labels saved to data/processed/")

if __name__ == "__main__":
    main()