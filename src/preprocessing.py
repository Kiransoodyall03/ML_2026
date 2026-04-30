# Define the 14 continuous numeric columns (Signal A through N)
# Assuming your pandas DataFrame columns are named 'Signal_A', 'Signal_B', etc.
sensor_columns = [f'Signal_{chr(i)}' for i in range(65, 79)] 

# Step 1: Scale the 14 continuous numeric columns
feature_scaler = ColumnTransformer(
    transformers=[
        ('standardize', StandardScaler(), sensor_columns)
    ],
    remainder='drop' # We drop the Time_Step and Sample_ID for the actual math
)

# Step 2: Build the master Preprocessing Pipeline
# This chains your custom imputer, the scaler, and dimensionality reduction.
preprocessing_pipeline = Pipeline(steps=[
    ('missing_data_imputer', SequentialSensorImputer(method='ffill', group_col='Sample_ID')),
    ('scaler', feature_scaler),
    # PCA to resolve hardware cross-talk and signal entanglement.
    # We will start with 8 components, but you should tune this hyperparameter later.
    ('dimensionality_reduction', PCA(n_components=8)) 
])