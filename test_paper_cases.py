import pandas as pd
import numpy as np
from config import Config
from data_loader import DataLoader
from stage1_prediction import RiskPredictor
from stage2_evaluation import RiskQuantifier
from stage3_optimization import LoanPortfolioOptimizer

def test_paper_cases():
    print("=== Testing Specific Paper Cases ===")
    
    # 1. Setup and Train Model (using original data to get a valid model)
    print("Initializing and training model on base dataset...")
    cfg = Config()
    loader = DataLoader(cfg)
    
    # Load original data to train the model
    try:
        df_full = loader.load_and_preprocess()
        X_train, y_train, _ = loader.split_data(df_full)
        
        predictor = RiskPredictor()
        predictor.train(X_train, y_train)
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    # 2. Construct Paper Cases Data
    # Data from user request
    # Columns map to: Current Ratio, Net Profit Margin, Asset Turnover, OCF/Sales,    # Paper companies data
    companies_data = {
        'Name': ['Apple Inc.', 'Walmart Inc.', 'NVIDIA Corp', 'Xerox Holdings', 'EYRC Worldwide'],
        'Sector': ['Technology', 'Consumer Services', 'Technology', 'Technology', 'Technology'],
        'currentRatio': [1.11, 0.86, 6.29, 1.50, 1.25],
        'netProfitMargin': [0.23, 0.03, 0.22, -0.04, -0.01],
        'assetTurnover': [0.80, 2.45, 0.54, 0.60, 2.55],
        'operatingCashFlowSalesRatio': [0.35, 0.06, 0.40, 0.07, 0.01],
        'debtEquityRatio': [0.59, 0.60, 0.20, 2.78, -4.19]
    }
    
    df_paper = pd.DataFrame(companies_data)
    
    # Handle Sector Encoding
    # We need the label encoder from the data loader to match the training data
    if hasattr(loader, 'label_encoder') and loader.label_encoder is not None:
        # Check if 'Sector' column exists in companies_data
        # It does.
        # Handle unseen labels if necessary (though these are common sectors)
        # For safety, we can check classes
        known_classes = set(loader.label_encoder.classes_)
        df_paper['Sector'] = df_paper['Sector'].apply(lambda x: x if x in known_classes else 'Unknown')
        
        # If 'Unknown' is not in classes, we might have an issue. 
        # But 'Technology' and 'Consumer Services' should be there.
        try:
            df_paper['sector_encoded'] = loader.label_encoder.transform(df_paper['Sector'])
        except ValueError as e:
            print(f"Warning: Sector encoding failed ({e}). Using 0.")
            df_paper['sector_encoded'] = 0
    else:
        df_paper['sector_encoded'] = 0
    
    print("Input Data (From Paper):")
    print("\nPaper Cases Data:")
    print(df_paper)
    
    # 3. Stage 1: Prediction
    print("\n--- Stage 1: Prediction ---")
    # Extract features matching the model's    # Extract features
    feature_cols = list(cfg.FEATURE_MAP.values())
    X_paper = df_paper[feature_cols].copy()
    X_paper['sector'] = df_paper['sector_encoded']
    
    pd_values = predictor.predict_pd(X_paper)
    df_paper['PD'] = pd_values
    
    print("Predicted Probability of Default (PD):")

    print(df_paper[['Name', 'PD']])

    # 4. Stage 2: Risk Quantification
    print("\n--- Stage 2: Risk Quantification ---")
    quantifier = RiskQuantifier(cfg)
    # Note: EWM/TOPSIS usually needs a larger dataset for relative comparison. 
    # Running it on just 5 companies compares them ONLY against each other.
    # ideally we should project them into the larger dataset's space, but for this test we compare them as a group.
    ri_scores = quantifier.calculate_risk_index(df_paper, pd_values)
    df_paper['RI'] = ri_scores
    
    print("Risk Index (RI):")
    print(df_paper[['Name', 'PD', 'RI']])

    # 5. Stage 3: Optimization
    print("\n--- Stage 3: Portfolio Optimization ---")
    # We need to set a budget suitable for 5 companies. 
    # Original budget 100M, max single 10M. 
    # Let's keep the same constraints.
    
    optimizer = LoanPortfolioOptimizer(df_paper, ri_scores, pd_values, cfg)
    pareto_solutions = optimizer.run_sa_na()
    
    print(f"\nFound {len(pareto_solutions)} Pareto optimal solutions.")
    
    if pareto_solutions:
        best_raroc_sol = min(pareto_solutions, key=lambda x: x[1][0])
        max_raroc = -best_raroc_sol[1][0]
        cvar = best_raroc_sol[1][1]
        
        print(f"\nBest RAROC Solution:")
        print(f"  RAROC: {max_raroc:.4f}")
        print(f"  CVaR:  {cvar:.4f}")
        
        # Show allocation
        solution_details = best_raroc_sol[0]
        print("\nAllocation Details:")
        for i, (amount, rate, decision) in enumerate(solution_details):
            company = df_paper.iloc[i]['Name']
            status = "Approved" if decision == 1 else "Rejected"
            print(f"  {company}: {status}, Amount: {amount:,.2f}, Rate: {rate:.2%}")

if __name__ == "__main__":
    test_paper_cases()
