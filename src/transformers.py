import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SequentialSensorImputer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to handle contiguous missing data blocks
    in time-series sensor data by grouping by Sample_ID.
    """
    def __init__(self, method='ffill', group_col='Sample_ID'):
        self.method = method
        self.group_col = group_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 1. Work on a copy
        X_imputed = X.copy()
        
        # 2. Identify the signal columns (we only want to impute the sensor data)
        # and exclude the Sample_ID and Time_Step
        signal_cols = [c for c in X.columns if 'signal' in c.lower()]
        
        # 3. Apply imputation within each Sample_ID group
        if self.method == 'ffill':
            # We group by Sample_ID and only fill the signal columns
            # .bfill() at the end handles cases where the first reading of a sample is NaN
            X_imputed[signal_cols] = X_imputed.groupby(self.group_col)[signal_cols].ffill().bfill()
        
        elif self.method == 'interpolate':
            # Linear interpolation within each block
            # This is more computationally expensive but smoother
            X_imputed[signal_cols] = X_imputed.groupby(self.group_col)[signal_cols].transform(
                lambda x: x.interpolate(method='linear').bfill().ffill()
            )
            
        # 4. Return the full DataFrame (with Sample_ID intact)
        return X_imputed