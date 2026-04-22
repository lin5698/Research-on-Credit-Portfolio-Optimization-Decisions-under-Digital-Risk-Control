"""
Model Optimization Script
Performs hyperparameter tuning for XGBoost and evaluates model performance
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import xgboost as xgb
from config import Config
from data_loader import DataLoader

def optimize_xgboost():
    """
    Perform grid search to find optimal XGBoost hyperparameters
    """
    print("=== XGBoost Hyperparameter Optimization ===\n")
    
    # Load data
    cfg = Config()
    loader = DataLoader(cfg)
    df = loader.load_and_preprocess()
    X_train, y_train, _ = loader.split_data(df)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Class distribution:\n{pd.Series(y_train).value_counts()}\n")
    
    # Define parameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Base model
    base_model = xgb.XGBClassifier(
        objective='multi:softprob',
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    # Grid search with cross-validation
    print("Starting Grid Search (this may take a while)...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,  # 3-fold CV due to small dataset
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2
    )
    
    # Fit
    grid_search.fit(X_train, y_train)
    
    # Results
    print("\n=== Optimization Results ===")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")
    
    # Evaluate best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_train)
    y_proba = best_model.predict_proba(X_train)
    
    print("\n=== Training Set Performance ===")
    print(classification_report(y_train, y_pred))
    
    # Calculate AUC (one-vs-rest for multiclass)
    try:
        auc = roc_auc_score(y_train, y_proba, multi_class='ovr', average='weighted')
        print(f"Weighted AUC: {auc:.4f}")
    except Exception as e:
        print(f"Could not calculate AUC: {e}")
    
    # Save best parameters to file
    with open('best_xgboost_params.txt', 'w') as f:
        f.write("Best XGBoost Parameters:\n")
        for param, value in grid_search.best_params_.items():
            f.write(f"{param}: {value}\n")
        f.write(f"\nBest CV F1 Score: {grid_search.best_score_:.4f}\n")
    
    print("\nBest parameters saved to 'best_xgboost_params.txt'")
    
    return grid_search.best_params_

if __name__ == "__main__":
    best_params = optimize_xgboost()
