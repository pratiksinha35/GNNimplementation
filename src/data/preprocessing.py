import deepchem as dc
import numpy as np

class DescriptorCleanerTransformer(dc.trans.Transformer):
    """Transformer to remove NaN/Inf columns"""
    
    def __init__(self):
        super().__init__(transform_X=True, transform_y=False, transform_w=False)
        self.columns_to_keep = None
        
    def transform(self, dataset, parallel=False):
        """Transform a Dataset object - REQUIRED for DeepChem compatibility"""
        if self.columns_to_keep is None:
            raise ValueError("Must call fit() before transform()")
        
        X, y, w, ids = dataset.X, dataset.y, dataset.w, dataset.ids
        
        # Select only valid columns
        X_clean = X[:, self.columns_to_keep]
        
        return dc.data.NumpyDataset(X_clean, y, w, ids)
    
    def fit(self, dataset):
        """Identify columns to keep"""
        X = dataset.X
        
        # Replace Inf with NaN
        X = np.where(np.isinf(X), np.nan, X)
        
        # Calculate NaN statistics per column
        nan_ratio = np.sum(np.isnan(X), axis=0) / len(X)
        
        # Keep only columns with exactly 0 NaN values
        self.columns_to_keep = np.where(nan_ratio == 0)[0]
        columns_to_drop = np.where(nan_ratio > 0)[0]
        
        print(f"Descriptor analysis:")
        print(f"  - Original shape: {X.shape}")
        print(f"  - Columns to keep: {len(self.columns_to_keep)}")
        print(f"  - Columns to drop: {len(columns_to_drop)}")
        
        if len(columns_to_drop) > 0:
            print(f"  - Dropped column indices: {columns_to_drop.tolist()}")
        
        return self