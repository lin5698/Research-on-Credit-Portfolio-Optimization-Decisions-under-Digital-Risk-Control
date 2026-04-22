"""
Advanced Parameter Optimization Script
Performs RandomizedSearchCV for XGBoost with Sector Awareness
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import xgboost as xgb
from config import Config
from data_loader import DataLoader
from sklearn.preprocessing import LabelEncoder

def optimize_xgboost_advanced():
    """
    Perform Randomized Search to find optimal XGBoost hyperparameters
    """
    print("=== Advanced XGBoost Hyperparameter Optimization ===\n")
    
    # Load data
    cfg = Config()
    loader = DataLoader(cfg)
    df = loader.load_and_preprocess()
    
    # Ensure sector encoding is present
    if 'Sector' in df.columns and 'sector_encoded' not in df.columns:
        le = LabelEncoder()
        df['Sector'] = df['Sector'].fillna('Unknown')
        df['sector_encoded'] = le.fit_transform(df['Sector'])
    
    # Split data (reproduce logic to get training set)
    # We need to manually handle the split to ensure we have the sector feature
    # loader.split_data returns X_train, y_train, optimize_df
    # But we want to ensure X_train has 'sector'
    
    X_train, y_train, _ = loader.split_data(df)
    
    # Check if sector is in X_train, if not add it from df
    if 'sector' not in X_train.columns and 'sector_encoded' in df.columns:
        # Map back using index
        X_train = X_train.copy()
        X_train['sector'] = df.loc[X_train.index, 'sector_encoded']
        print("Added 'sector' feature to training data.")
    
    print(f"Training set size: {len(X_train)}")
    print(f"Features: {X_train.columns.tolist()}")
    
    # Define parameter distribution
    param_dist = {
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
        'n_estimators': [100, 200, 300, 400, 500],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [0, 0.01, 0.1, 1]
    }
    
    # Base model
    base_model = xgb.XGBClassifier(
        objective='multi:softprob',
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    # Randomized Search
    print("Starting Randomized Search (50 iterations)...")
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=50,
        scoring='f1_weighted',
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Fit
    # Compute sample weights for balance
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    random_search.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Results
    print("\n=== Optimization Results ===")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best CV F1 Score: {random_search.best_score_:.4f}")
    
    # Evaluate best model
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_train)
    y_proba = best_model.predict_proba(X_train)
    
    print("\n=== Training Set Performance ===")
    print(classification_report(y_train, y_pred))
    
    try:
        auc = roc_auc_score(y_train, y_proba, multi_class='ovr', average='weighted')
        print(f"Weighted AUC: {auc:.4f}")
    except Exception as e:
        print(f"Could not calculate AUC: {e}")
        
    # Save best parameters
    with open('best_xgboost_params_advanced.txt', 'w') as f:
        f.write("Best XGBoost Parameters (Advanced):\n")
        for param, value in random_search.best_params_.items():
            f.write(f"{param}: {value}\n")
        f.write(f"\nBest CV F1 Score: {random_search.best_score_:.4f}\n")
        
    return random_search.best_params_

if __name__ == "__main__":
    optimize_xgboost_advanced()
