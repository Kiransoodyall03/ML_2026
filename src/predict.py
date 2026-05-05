import torch
import pandas as pd
import numpy as np
import os
from transformers import SequentialSensorImputer
from train_resnet import SensorResNet # Imports your exact architecture

def main():
    print("\n=== Final Phase: Generating Kaggle Submission ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the Trained Model
    print("1. Loading ResNet model...")
    model = SensorResNet(num_classes=6).to(device)
    # Load the weights we just saved
    model.load_state_dict(torch.load("data/processed/resnet_weights.pth", map_location=device))
    model.eval() # Set to evaluation mode

    # 2. Load and Impute the Hidden Test Data
    print("2. Loading and Imputing Test Data...")
    df_test = pd.read_parquet("data/raw/test_data.parquet", engine='pyarrow')
    if 'Sample_ID' not in df_test.columns:
        df_test = df_test.reset_index()

    imputer = SequentialSensorImputer(method='interpolate', group_col='Sample_ID')
    df_test_imp = imputer.fit_transform(df_test)

    # 3. Reshape and Predict
    print("3. Running Deep Learning Inference...")
    sample_ids = df_test_imp['Sample_ID'].unique()
    signal_cols = [c for c in df_test_imp.columns if 'signal' in c.lower()]
    grouped = df_test_imp.groupby('Sample_ID')
    
    predictions = []

    with torch.no_grad(): # Don't track gradients (saves memory/time)
        for s_id in sample_ids:
            # Extract the 100 rows for this specific sample
            sample_data = grouped.get_group(s_id)[signal_cols].values
            
            # Reshape to (Channels, Time_Steps) -> (14, 100) and add Batch dimension
            tensor_data = torch.tensor(sample_data.T, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Feed to model
            output = model(tensor_data)
            pred_class = torch.argmax(output, dim=1).item()
            
            # PyTorch outputs 0-5, but Kaggle expects 1-6. We must add 1 back!
            predictions.append({'Sample_ID': s_id, 'class_label': pred_class + 1})

    # 4. Save Submission
    print("4. Saving submission.csv...")
    sub_df = pd.DataFrame(predictions)
    sub_df.to_csv("submission.csv", index=False)
    
    print("-" * 40)
    print("Success! 'submission.csv' has been generated in your root folder.")
    print("-" * 40)

if __name__ == "__main__":
    main()