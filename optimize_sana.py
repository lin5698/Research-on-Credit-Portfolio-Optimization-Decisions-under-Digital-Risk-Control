"""
SA-NA Parameter Optimization Script
Tests different temperature and cooling rate combinations to find optimal settings
"""
import numpy as np
import pandas as pd
from config import Config
from data_loader import DataLoader
from stage1_prediction import RiskPredictor
from stage2_evaluation import RiskQuantifier
from stage3_optimization import LoanPortfolioOptimizer
import time

def optimize_sana_params():
    print("=== SA-NA Parameter Optimization ===\n")
    
    # Setup environment
    cfg = Config()
    loader = DataLoader(cfg)
    try:
        df_full = loader.load_and_preprocess()
        X_train, y_train, df_optimize = loader.split_data(df_full)
    except Exception as e:
        print(f"Data Error: {e}")
        return

    # Train model (quick train)
    predictor = RiskPredictor()
    predictor.train(X_train, y_train)
    
    # Prepare optimization data
    feature_cols = list(cfg.FEATURE_MAP.values())
    X_opt = df_optimize[feature_cols].copy()
    if 'sector_encoded' in df_optimize.columns:
        X_opt['sector'] = df_optimize['sector_encoded']
    elif 'sector' in X_train.columns:
         # Fallback if sector missing in df_optimize but present in train
         pass
         
    pd_values = predictor.predict_pd(X_opt)
    quantifier = RiskQuantifier(cfg)
    ri_scores = quantifier.calculate_risk_index(df_optimize, pd_values)
    
    # Parameter Grid
    temps = [500, 1000, 2000]
    cooling_rates = [0.90, 0.95, 0.98]
    
    results = []
    
    print(f"{'Temp':<10} {'Cooling':<10} {'Pareto Size':<15} {'Best RAROC':<15} {'Time (s)':<10}")
    print("-" * 65)
    
    for temp in temps:
        for rate in cooling_rates:
            # Update config temporarily
            cfg.SA_TEMP_INIT = temp
            cfg.SA_COOLING_RATE = rate
            
            start_time = time.time()
            optimizer = LoanPortfolioOptimizer(df_optimize, ri_scores, pd_values, cfg)
            # Suppress print output from optimizer
            import sys, os
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            try:
                pareto_solutions = optimizer.run_sa_na()
            finally:
                sys.stdout = original_stdout
            
            elapsed = time.time() - start_time
            
            if pareto_solutions:
                best_raroc = -min(pareto_solutions, key=lambda x: x[1][0])[1][0]
                pareto_size = len(pareto_solutions)
            else:
                best_raroc = 0
                pareto_size = 0
                
            print(f"{temp:<10} {rate:<10.2f} {pareto_size:<15} {best_raroc:<15.4f} {elapsed:<10.2f}")
            results.append({
                'temp': temp,
                'rate': rate,
                'pareto_size': pareto_size,
                'best_raroc': best_raroc,
                'time': elapsed
            })
            
    # Find best config
    best_result = max(results, key=lambda x: x['best_raroc'])
    print("\n=== Best SA-NA Parameters ===")
    print(f"Temperature: {best_result['temp']}")
    print(f"Cooling Rate: {best_result['rate']}")
    print(f"Achieved RAROC: {best_result['best_raroc']:.4f}")
    
    # Save to file
    with open('best_sana_params.txt', 'w') as f:
        f.write(f"SA_TEMP_INIT={best_result['temp']}\n")
        f.write(f"SA_COOLING_RATE={best_result['rate']}\n")
        
    return best_result

if __name__ == "__main__":
    optimize_sana_params()
