"""
Detailed Analysis of Paper Case Companies
Calculates all relevant parameters and compares with provided data
"""
import pandas as pd
import numpy as np
from config import Config
from data_loader import DataLoader
from stage1_prediction import RiskPredictor
from stage2_evaluation import RiskQuantifier

def analyze_paper_companies():
    print("=== Detailed Analysis of Paper Case Companies ===\n")
    
    # Setup
    cfg = Config()
    loader = DataLoader(cfg)
    
    # Load and train model on full dataset
    df_full = loader.load_and_preprocess()
    X_train, y_train, _ = loader.split_data(df_full)
    
    predictor = RiskPredictor()
    predictor.train(X_train, y_train)
    
    # Paper companies data
    companies_data = {
        'Name': ['Apple Inc.', 'Walmart Inc.', 'NVIDIA Corp', 'Xerox Holdings', 'EYRC Worldwide'],
        'currentRatio': [1.11, 0.86, 6.29, 1.50, 1.25],
        'netProfitMargin': [0.23, 0.03, 0.22, -0.04, -0.01],
        'assetTurnover': [0.80, 2.45, 0.54, 0.60, 2.55],
        'operatingCashFlowSalesRatio': [0.35, 0.06, 0.40, 0.07, 0.01],
        'debtEquityRatio': [0.59, 0.60, 0.20, 2.78, -4.19]
    }
    
    df_paper = pd.DataFrame(companies_data)
    
    print("Input Data (From Paper):")
    print("=" * 100)
    print(df_paper.to_string(index=False))
    print("\n")
    
    # Extract features
    feature_cols = list(cfg.FEATURE_MAP.values())
    X_paper = df_paper[feature_cols]
    
    # Stage 1: Prediction
    print("Stage 1: Risk Prediction (XGBoost + Calibration)")
    print("=" * 100)
    pd_values = predictor.predict_pd(X_paper)
    df_paper['PD'] = pd_values
    
    # Get class predictions
    if predictor.calibrated_model:
        y_pred = predictor.calibrated_model.predict(X_paper)
    else:
        y_pred = predictor.model.predict(X_paper)
    
    rating_reverse_map = {v: k for k, v in cfg.RATING_MAP.items()}
    rating_labels = ['A (Low Risk)', 'BBB (Medium-Low)', 'BB (Medium-High)', 'B-D (High Risk)']
    
    for i, row in df_paper.iterrows():
        pred_class = y_pred[i]
        print(f"{row['Name']:20s} | PD: {row['PD']:.4f} ({row['PD']*100:.2f}%) | Predicted Class: {pred_class} ({rating_labels[pred_class]})")
    
    print("\n")
    
    # Stage 2: Risk Quantification
    print("Stage 2: Risk Quantification (EWM-TOPSIS)")
    print("=" * 100)
    quantifier = RiskQuantifier(cfg)
    ri_scores = quantifier.calculate_risk_index(df_paper, pd_values)
    df_paper['RI'] = ri_scores
    
    # Calculate intermediate scores for transparency
    # Level 1: Financial Strength
    data_matrix = df_paper[feature_cols].values
    is_benefit = [True, True, True, True, False]
    w1 = quantifier.entropy_weight(data_matrix)
    strength_score = quantifier.topsis(data_matrix, w1, is_benefit)
    df_paper['Strength_Score'] = strength_score
    
    # Level 2 weights
    rating_score = 1 - pd_values
    secondary_matrix = np.column_stack((strength_score, rating_score))
    w2 = quantifier.entropy_weight(secondary_matrix)
    
    print(f"Level 1 Weights (Financial Indicators):")
    for i, (key, col) in enumerate(cfg.FEATURE_MAP.items()):
        print(f"  {key:25s}: {w1[i]:.4f}")
    
    print(f"\nLevel 2 Weights:")
    print(f"  Strength Score: {w2[0]:.4f}")
    print(f"  Credit Rating:  {w2[1]:.4f}")
    
    print("\nDetailed Results:")
    print("-" * 100)
    for i, row in df_paper.iterrows():
        print(f"{row['Name']:20s} | Strength: {row['Strength_Score']:.4f} | PD: {row['PD']:.4f} | RI: {row['RI']:.2f}")
    
    print("\n")
    
    # Comparison with paper data
    print("Comparison with Paper Data")
    print("=" * 100)
    print(f"{'Company':<20s} {'Input Match':<15s} {'Model Output':<50s}")
    print("-" * 100)
    
    for i, row in df_paper.iterrows():
        input_vals = f"L={row['currentRatio']:.2f}, I={row['netProfitMargin']:.2f}, C={row['assetTurnover']:.2f}, S={row['operatingCashFlowSalesRatio']:.2f}, E={row['debtEquityRatio']:.2f}"
        output_vals = f"PD={row['PD']:.4f}, RI={row['RI']:.2f}, Strength={row['Strength_Score']:.4f}"
        print(f"{row['Name']:<20s} {'✓':<15s} {output_vals}")
    
    print("\n")
    
    # Summary statistics
    print("Summary Statistics")
    print("=" * 100)
    print(f"Average PD:       {df_paper['PD'].mean():.4f}")
    print(f"Average RI:       {df_paper['RI'].mean():.2f}")
    print(f"Lowest Risk:      {df_paper.loc[df_paper['RI'].idxmin(), 'Name']} (RI={df_paper['RI'].min():.2f})")
    print(f"Highest Risk:     {df_paper.loc[df_paper['RI'].idxmax(), 'Name']} (RI={df_paper['RI'].max():.2f})")
    
    # Save detailed results
    output_file = 'paper_companies_analysis.csv'
    df_paper.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    return df_paper

if __name__ == "__main__":
    results = analyze_paper_companies()
