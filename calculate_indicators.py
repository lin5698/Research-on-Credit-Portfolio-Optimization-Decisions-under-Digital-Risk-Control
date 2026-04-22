"""
Calculate Quantitative Indicators and Descriptive Statistics
For the 123 "Known" Enterprises (Training Set)
"""
import pandas as pd
import numpy as np
from config import Config
from data_loader import DataLoader

def calculate_and_describe():
    print("=== Indicator Calculation & Descriptive Statistics ===\n")
    
    cfg = Config()
    loader = DataLoader(cfg)
    
    # Load full dataset
    df = loader.load_and_preprocess()
    
    # Split to get the 123 training samples (before SMOTE)
    # We need to access the internal split logic or just reproduce it
    # The loader.split_data returns X_res (SMOTE), y_res, optimize_df
    # We want the original 123 training samples.
    
    # Let's peek at how split_data works in data_loader.py
    # It uses: train_df = df.sample(n=123, random_state=42)
    
    n_train = 123
    # Ensure reproducibility matches data_loader.py
    train_df = df.sample(n=n_train, random_state=42)
    
    print(f"Selected {len(train_df)} enterprises for analysis (Training Set).")
    
    # Select the 5 core indicators
    # Mapping from Config:
    # 'Activity_Level': 'currentRatio'
    # 'Sales_Capability': 'netProfitMargin'
    # 'Cooperation_Capability': 'assetTurnover'
    # 'Marketing_Capability': 'operatingCashFlowSalesRatio'
    # 'Return_Rate': 'debtEquityRatio'
    
    indicators = {
        'currentRatio': '活力程度(Li)',
        'netProfitMargin': '销售能力(Ii)',
        'assetTurnover': '合作能力(Ci)',
        'operatingCashFlowSalesRatio': '营销能力(Si)',
        'debtEquityRatio': '被退货率(Ei)' # Potential Risk Rate
    }
    
    # Extract data
    df_indicators = train_df[list(indicators.keys())].copy()
    df_indicators.rename(columns=indicators, inplace=True)
    
    # Add Company Name if available (it is in df)
    if 'Name' in train_df.columns:
        df_indicators.insert(0, '企业名称', train_df['Name'])
    
    # 1. Descriptive Statistics
    print("\n--- Descriptive Statistics ---")
    stats = df_indicators.describe().T
    stats = stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    print(stats)
    
    # Save statistics
    stats.to_csv('indicator_statistics.csv')
    print("\nStatistics saved to: indicator_statistics.csv")
    
    # 2. Table 2 Data (First 10 rows as sample)
    print("\n--- Table 2: Enterprise Indicators (Sample) ---")
    print(df_indicators.head(10).to_string(index=False))
    
    # Save full table
    df_indicators.to_csv('table2_enterprise_indicators.csv', index=False)
    print("\nFull Table 2 data saved to: table2_enterprise_indicators.csv")
    
    return df_indicators, stats

if __name__ == "__main__":
    calculate_and_describe()
