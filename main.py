import pandas as pd
from config import Config
from data_loader import DataLoader
from stage1_prediction import RiskPredictor
from stage2_evaluation import RiskQuantifier
from stage3_optimization import LoanPortfolioOptimizer

def main():
    print("=== Credit Risk Portfolio Optimization System ===")
    cfg = Config()
    loader = DataLoader(cfg)
  
    # 1. Data Loading and Splitting
    try:
        df_full = loader.load_and_preprocess()
        X_train, y_train, df_optimize = loader.split_data(df_full)
        print(f"Training set: {X_train.shape}, Optimization set: {df_optimize.shape}")
    except Exception as e:
        print(f"Error in Data Loading: {e}")
        return
  
    # 2. Stage 1: XGBoost-SHAP Prediction
    print("Stage 1: Training XGBoost Model...")
    predictor = RiskPredictor()
    # Note: X_train already has 'sector' if data_loader added it
    predictor.train(X_train, y_train)
    
    print("Stage 1: Generating SHAP Explanations...")
    # Use a subset for SHAP to save time
    shap_sample = X_train.sample(min(100, len(X_train)), random_state=42)
    predictor.explain_model(shap_sample)
    
    # Predict PD for Optimization Set
    # We need to ensure optimize_df has the same features as X_train
    # data_loader.split_data returns optimize_df with ALL columns (including features and target)
    # We need to extract features + sector
    feature_cols = list(cfg.FEATURE_MAP.values())
    X_opt = df_optimize[feature_cols].copy()
    
    # Add sector if it exists in optimize_df
    if 'sector_encoded' in df_optimize.columns:
        X_opt['sector'] = df_optimize['sector_encoded']
    elif 'sector' in X_train.columns:
        # If X_train has sector but optimize_df doesn't (unlikely if split from same df), handle it
        # But data_loader adds sector_encoded to the whole df before split
        pass
        
    pd_values = predictor.predict_pd(X_opt)
    df_optimize['PD'] = pd_values
    
    # Generate SHAP explanations for optimization set (optional, can be resource intensive)
    # predictor.explain_model(X_opt) # Commented out to avoid redundant SHAP generation if not needed
  
    # 3. Stage 2: EWM-TOPSIS Risk Quantification
    quantifier = RiskQuantifier(cfg)
    ri_scores = quantifier.calculate_risk_index(df_optimize, pd_values)
  
    # Merge results for viewing
    df_optimize = df_optimize.copy()
    df_optimize['PD'] = pd_values
    df_optimize['RI'] = ri_scores
    
    print("\nSample Risk Assessment (Top 5):")
    print(df_optimize[['PD', 'RI']].head())
  
    # 4. Stage 3: SA-NA Portfolio Optimization
    optimizer = LoanPortfolioOptimizer(df_optimize, ri_scores, pd_values, cfg)
    pareto_solutions = optimizer.run_sa_na()
  
    print(f"\nOptimization Complete. Found {len(pareto_solutions)} Pareto optimal solutions.")
  
    if not pareto_solutions:
        print("No valid solutions found.")
        return

    # Display Best Solution (Max RAROC)
    # pareto_solutions is list of (solution, objectives)
    # objectives = (-RAROC, CVaR)
    # So min of objective[0] is max RAROC
    best_raroc_sol = min(pareto_solutions, key=lambda x: x[1][0]) 
    
    max_raroc = -best_raroc_sol[1][0]
    cvar_at_max_raroc = best_raroc_sol[1][1]
    
    print(f"\nBest RAROC Solution:")
    print(f"  RAROC: {max_raroc:.4f}")
    print(f"  CVaR:  {cvar_at_max_raroc:.4f}")
    
    # Analyze the portfolio
    solution_details = best_raroc_sol[0] # List of (amount, rate, decision)
    amounts = [s[0] for s in solution_details]
    decisions = [s[2] for s in solution_details]
    
    total_invested = sum(a * d for a, d in zip(amounts, decisions))
    num_companies = sum(decisions)
    
    print(f"  Total Invested: {total_invested:,.2f}")
    print(f"  Number of Companies Funded: {num_companies}")

if __name__ == "__main__":
    main()
