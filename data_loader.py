import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from config import Config

class DataLoader:
    def __init__(self, config):
        self.cfg = config
        self.label_encoder = None

    def load_and_preprocess(self):
        """
        Load data, handle missing values, map ratings, and apply winsorization.
        """
        print("Loading and preprocessing data...")
        df = pd.read_csv(self.cfg.DATA_PATH)
      
        # 1. Missing Value Handling
        # Drop columns with > 20% missing values
        df = df.dropna(thresh=len(df)*0.8, axis=1)
        # Fill remaining missing values with mean (numeric only)
        df = df.fillna(df.mean(numeric_only=True))
      
        # 2. Rating Mapping
        if 'Rating' not in df.columns:
            raise ValueError("Column 'Rating' not found in dataset")
            
        df['target'] = df['Rating'].map(self.cfg.RATING_MAP)
        # Drop rows where target is NaN (if any unmapped ratings exist)
        df = df.dropna(subset=['target'])
        df['target'] = df['target'].astype(int)
      
        # 3. Winsorization (1% level)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Exclude target from winsorization if it's numeric
        numeric_cols = [c for c in numeric_cols if c != 'target']
        
        for col in numeric_cols:
            df[col] = winsorize(df[col], limits=[0.01, 0.01])
            
        # 4. Sector Encoding (New Feature)
        if 'Sector' in df.columns:
            self.label_encoder = LabelEncoder()
            # Fill NaN sectors with 'Unknown'
            df['Sector'] = df['Sector'].fillna('Unknown')
            df['sector_encoded'] = self.label_encoder.fit_transform(df['Sector'])
            print(f"Sector encoding applied. {len(self.label_encoder.classes_)} sectors found.")
        else:
            print("Warning: 'Sector' column not found. Skipping sector encoding.")
            df['sector_encoded'] = 0
          
        print(f"Data loaded: {df.shape}")
        return df

    def split_data(self, df):
        """
        Split data into training set (for XGBoost) and optimization set (for Portfolio Optimization).
        Applies SMOTE to the training set.
        """
        # Paper setting: 123 known for training, 32 unknown for optimization
        # We will try to match this ratio or just use a random split if dataset size differs
        
        n_total = len(df)
        n_train = 123
        n_opt = 32
        
        if n_total < (n_train + n_opt):
            # If dataset is smaller, just split 80/20
            train_df = df.sample(frac=0.8, random_state=42)
            optimize_df = df.drop(train_df.index)
        else:
            train_df = df.sample(n=n_train, random_state=42)
            # Ensure optimize_df doesn't overlap
            remaining = df.drop(train_df.index)
            optimize_df = remaining.sample(n=min(n_opt, len(remaining)), random_state=42)
      
        # Extract features and target for training
        # Ensure all feature columns exist
        feature_cols = list(self.cfg.FEATURE_MAP.values())
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        X_train = train_df[feature_cols].copy()
        
        # Add sector feature if available
        if 'sector_encoded' in train_df.columns:
            X_train['sector'] = train_df['sector_encoded']
            print("Added 'sector' feature to training data.")
            
        y_train = train_df['target']
      
        # SMOTE Data Augmentation
        # Check if we have enough samples per class for SMOTE
        # SMOTE requires at least k_neighbors + 1 samples in the minority class
        min_class_samples = y_train.value_counts().min()
        k_neighbors = 5
        
        if min_class_samples > k_neighbors:
            print("Applying SMOTE...")
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_res, y_res = smote.fit_resample(X_train, y_train)
        else:
            print(f"Warning: Not enough samples for SMOTE (min class: {min_class_samples}). Using original data.")
            X_res, y_res = X_train, y_train
      
        return X_res, y_res, optimize_df
