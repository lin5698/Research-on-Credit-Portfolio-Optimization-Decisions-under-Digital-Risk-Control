"""
Match user-provided company data to dataset entries
Finds the closest matching companies in the dataset based on financial indicators
"""
import pandas as pd
import numpy as np
from config import Config
from data_loader import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def match_companies():
    print("=== Company Matching Analysis ===\n")
    
    cfg = Config()
    loader = DataLoader(cfg)
    df = loader.load_and_preprocess()
    
    # User provided data (from previous context)
    user_companies = {
        'Apple Inc.': [1.11, 0.23, 0.80, 0.35, 0.59],
        'Walmart Inc.': [0.86, 0.03, 2.45, 0.06, 0.60],
        'NVIDIA Corp': [6.29, 0.22, 0.54, 0.40, 0.20],
        'Xerox Holdings': [1.50, -0.04, 0.60, 0.07, 2.78],
        'EYRC Worldwide': [1.25, -0.01, 2.55, 0.01, -4.19]
    }
    
    feature_cols = list(cfg.FEATURE_MAP.values())
    # ['currentRatio', 'netProfitMargin', 'assetTurnover', 'operatingCashFlowSalesRatio', 'debtEquityRatio']
    
    print(f"Matching based on features: {feature_cols}\n")
    
    # Prepare dataset features
    X_dataset = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dataset)
    
    # Fit Nearest Neighbors
    nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn.fit(X_scaled)
    
    results = []
    
    for name, values in user_companies.items():
        print(f"Finding matches for: {name}")
        print(f"  Input: {values}")
        
        # Scale input
        input_scaled = scaler.transform([values])
        
        # Find neighbors
        distances, indices = nn.kneighbors(input_scaled)
        
        print(f"  Top 3 Matches in Dataset:")
        for i in range(3):
            idx = indices[0][i]
            dist = distances[0][i]
            match_row = df.iloc[idx]
            match_name = match_row['Name'] if 'Name' in df.columns else f"Index {idx}"
            
            # Calculate actual difference
            diff = np.abs(match_row[feature_cols].values - values)
            avg_diff = np.mean(diff)
            
            print(f"    {i+1}. {match_name:30s} (Dist: {dist:.4f}, Avg Diff: {avg_diff:.4f})")
            # Convert to float array for printing to avoid type errors
            match_values = match_row[feature_cols].values.astype(float)
            print(f"       Data: {np.round(match_values, 2)}")
            
        print("-" * 50)

if __name__ == "__main__":
    match_companies()
