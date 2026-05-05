import pandas as pd
import numpy as np
import xgboost as xgb 
from transformers import SequentialSensorImputer
from evaluation import evaluate_model_cv 

# --- Time Domain Features ---
def zero_crossing_rate(series):
    """Calculates how often the signal crosses the zero axis."""
    return ((series.shift(1) * series) < 0).sum()

# --- Frequency Domain Features (FFT) ---
def dominant_frequency(series):
    """Extracts the magnitude of the dominant frequency using FFT."""
    # Compute the Fourier Transform
    fft_mag = np.abs(np.fft.fft(series.values))
    n = len(series)
    if n < 2: return 0
    # Exclude DC component (0 Hz at index 0) and take the max of the first half
    return np.max(fft_mag[1:n//2])

def spectral_energy(series):
    """Calculates the total energy of the signal in the frequency domain."""
    fft_mag = np.abs(np.fft.fft(series.values))
    n = len(series)
    if n < 2: return 0
    # Sum of squared magnitudes
    return np.sum(fft_mag[1:n//2]**2) / n

def extract_features(df):
    """
    Advanced aggregation: turns 100 rows of sensor data into 
    descriptive physical and frequency features.
    """
    print("--- Extracting advanced time & frequency features...")
    
    sample_col = [c for c in df.columns if 'sample' in c.lower()][0]
    signal_cols = [c for c in df.columns if 'signal' in c.lower()]
    
    # NEW: Added dominant_frequency and spectral_energy to our aggregation
    features = df.groupby(sample_col)[signal_cols].agg([
        'mean', 'std', 'min', 'max', 'median', 
        zero_crossing_rate, 
        dominant_frequency, 
        spectral_energy
    ])
    
    # Energy Feature: Signal Magnitude Area (SMA)
    sma = df.groupby(sample_col)[signal_cols].apply(lambda x: x.abs().sum().sum() / 100)
    
    # Flatten column names
    features.columns = ['_'.join(col).strip() for col in features.columns.values]
    
    # Range Feature: Max - Min
    for sig in signal_cols:
        features[f'{sig}_range'] = features[f'{sig}_max'] - features[f'{sig}_min']
    
    # Attach SMA
    features['signal_energy_sma'] = sma
    
    return features.reset_index().rename(columns={sample_col: 'Sample_ID'})

def main():
    print("\n=== COMS3007A: Training XGBoost with FFT Features ===")
    
    data_path = "data/raw/train_data.parquet" 
    labels_path = "data/raw/train_labels.csv"
    
    print(f"1. Loading files...")
    df_data = pd.read_parquet(data_path, engine='pyarrow')
    df_labels = pd.read_csv(labels_path)
    
    if 'Sample_ID' not in df_data.columns:
        df_data = df_data.reset_index()

    print("2. Running imputation (Interpolate)...")
    imputer = SequentialSensorImputer(method='interpolate', group_col='Sample_ID')
    df_imputed = imputer.fit_transform(df_data)

    # 3. Feature Engineering (Now with FFT!)
    df_features = extract_features(df_imputed)
    
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
    
    y = y - 1 
    
    print(f"--- Feature set expanded: {X.shape[1]} features.")
    
# 4. XGBoost Model (Optimized via Optuna)
    model = xgb.XGBClassifier(
        n_estimators=353, 
        max_depth=6,
        learning_rate=0.1472,
        subsample=0.9552,
        colsample_bytree=0.6650,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    
    print("4. Starting Cross-Validation...")
    evaluate_model_cv(model, X, y, n_splits=5)

if __name__ == "__main__":
    main()