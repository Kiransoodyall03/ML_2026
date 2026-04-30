import pandas as pd
import numpy as np
import os

def generate_dummy_dataset(num_samples=50, output_dir="data/raw"):
    """
    Generates a mock sensor dataset simulating the COMS3007A assignment schema.
    Creates both a Parquet data file and a CSV labels file.
    """
    print(f"Generating {num_samples} mock samples...")
    
    # 1. Ensure the target directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Define the schema constraints
    time_steps_per_sample = 100
    signal_columns = [f'Signal_{chr(i)}' for i in range(65, 79)] # Signal_A through Signal_N
    target_classes = [1, 2, 3, 4, 5, 6] # The 6 physical actions
    
    data_rows = []
    labels = []
    
    # 3. Generate the data loop
    for sample_id in range(1, num_samples + 1):
        # Assign a random physical action (Class) for this Sample_ID
        assigned_class = np.random.choice(target_classes)
        labels.append({'Sample_ID': sample_id, 'Class': assigned_class})
        
        # Generate exactly 100 time steps for this sample
        for t in range(1, time_steps_per_sample + 1):
            row = {
                'Sample_ID': sample_id,
                'Time_Step': t
            }
            
            # Generate random continuous numeric data for the 14 signals
            for sig in signal_columns:
                # Introduce contiguous missing data blocks (simulating sensor dropout)
                # 5% chance a reading drops out and becomes NaN
                if np.random.rand() < 0.05:
                    row[sig] = np.nan
                else:
                    # Normal sensor reading (random float)
                    row[sig] = np.random.randn()
                    
            data_rows.append(row)
            
    # 4. Convert to pandas DataFrames
    df_data = pd.DataFrame(data_rows)
    df_labels = pd.DataFrame(labels)
    
    # 5. Save to disk in the required formats
    data_filepath = os.path.join(output_dir, "dummy_train_data.parquet")
    labels_filepath = os.path.join(output_dir, "dummy_train_labels.csv")
    
    # Note: Requires 'pyarrow' or 'fastparquet' installed (from our requirements.txt)
    df_data.to_parquet(data_filepath, engine='pyarrow', index=False)
    df_labels.to_csv(labels_filepath, index=False)
    
    print("-" * 30)
    print("Dummy Dataset Generation Complete!")
    print(f"Data Shape:   {df_data.shape} -> Saved to {data_filepath}")
    print(f"Labels Shape: {df_labels.shape} -> Saved to {labels_filepath}")
    print("-" * 30)

if __name__ == "__main__":
    # Generate 100 fake samples (100 samples * 100 rows = 10,000 rows of data)
    generate_dummy_dataset(num_samples=100)