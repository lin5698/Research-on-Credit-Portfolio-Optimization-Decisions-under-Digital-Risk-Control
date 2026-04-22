"""
Sector-Aware Model Training with Tech Industry Calibration
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
from config import Config
from data_loader import DataLoader
from stage1_prediction import RiskPredictor
from stage2_evaluation import RiskQuantifier

def train_sector_aware_model():
    """
    Train model with sector information to improve tech company predictions
    """
    print("=== Sector-Aware Model Training ===\n")
    
    cfg = Config()
    
    # Load full dataset with sector info
    df = pd.read_csv(cfg.DATA_PATH)
    print(f"Total samples: {len(df)}")
    
    # Preprocess
    loader = DataLoader(cfg)
    df = loader.load_and_preprocess()
    
    # Add sector encoding as additional feature
    if 'Sector' in df.columns:
        le = LabelEncoder()
        df['sector_encoded'] = le.fit_transform(df['Sector'].fillna('Unknown'))
        print(f"Sectors encoded: {len(le.classes_)} unique sectors")
        
        # Identify tech companies
        tech_mask = df['Sector'].str.contains('Technology|Telecom|IT', case=False, na=False)
        print(f"Tech companies in dataset: {tech_mask.sum()} ({tech_mask.sum()/len(df)*100:.1f}%)")
    else:
        print("Warning: No sector information available")
        df['sector_encoded'] = 0
    
    # Split data with stratification by sector
    X_train, y_train, df_opt = loader.split_data(df)
    
    # Add sector feature to training data
    if 'sector_encoded' in df.columns:
        # Get sector for training samples
        train_indices = X_train.index
        X_train_with_sector = X_train.copy()
        X_train_with_sector['sector'] = df.loc[train_indices, 'sector_encoded'].values
        
        print(f"\nTraining set: {len(X_train_with_sector)} samples")
        print(f"Features: {X_train_with_sector.columns.tolist()}")
    else:
        X_train_with_sector = X_train
    
    # Train model with sector feature
    print("\nTraining XGBoost with sector awareness...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        subsample=1.0,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    
    # Compute sample weights (emphasize tech companies if available)
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    model.fit(X_train_with_sector, y_train, sample_weight=sample_weights)
    
    # Feature importance
    print("\nFeature Importance:")
    feature_names = X_train_with_sector.columns
    importances = model.feature_importances_
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {name:30s}: {imp:.4f}")
    
    # Save model
    import pickle
    with open('sector_aware_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'label_encoder': le if 'Sector' in df.columns else None}, f)
    
    print("\nSector-aware model saved to: sector_aware_model.pkl")
    
    return model, le if 'Sector' in df.columns else None

def evaluate_tech_companies_with_calibration():
    """
    Re-evaluate tech companies using sector-aware model
    """
    print("\n" + "="*80)
    print("=== Re-evaluating Tech Companies with Calibrated Model ===")
    print("="*80 + "\n")
    
    # Load sector-aware model
    import pickle
    try:
        with open('sector_aware_model.pkl', 'rb') as f:
            saved = pickle.load(f)
            model = saved['model']
            le = saved['label_encoder']
        print("Loaded sector-aware model\n")
    except:
        print("Training new sector-aware model...")
        model, le = train_sector_aware_model()
    
    # Paper companies
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
    
    # Encode sectors
    if le is not None:
        df_paper['sector_encoded'] = le.transform(df_paper['Sector'])
    else:
        df_paper['sector_encoded'] = 0
    
    cfg = Config()
    feature_cols = list(cfg.FEATURE_MAP.values())
    X_paper = df_paper[feature_cols].copy()
    X_paper['sector'] = df_paper['sector_encoded'].values
    
    # Predict with sector-aware model
    y_pred = model.predict(X_paper)
    y_proba = model.predict_proba(X_paper)
    pd_values = y_proba[:, -1]  # High risk class probability
    
    df_paper['PD_Calibrated'] = pd_values
    df_paper['Predicted_Class'] = y_pred
    
    # Calculate Risk Index
    quantifier = RiskQuantifier(cfg)
    ri_scores = quantifier.calculate_risk_index(df_paper[feature_cols + ['Name']], pd_values)
    df_paper['RI_Calibrated'] = ri_scores
    
    # Display results
    print("Calibrated Results:")
    print("="*80)
    rating_labels = ['A (Low)', 'BBB (Med-Low)', 'BB (Med-High)', 'B-D (High)']
    
    for i, row in df_paper.iterrows():
        print(f"{row['Name']:20s} | Sector: {row['Sector']:20s}")
        print(f"  PD: {row['PD_Calibrated']:.4f} ({row['PD_Calibrated']*100:.2f}%)")
        print(f"  RI: {row['RI_Calibrated']:.2f}")
        print(f"  Class: {rating_labels[row['Predicted_Class']]}")
        print()
    
    # Save results
    df_paper.to_csv('tech_companies_calibrated_results.csv', index=False)
    print("Results saved to: tech_companies_calibrated_results.csv")
    
    return df_paper

if __name__ == "__main__":
    # Train sector-aware model
    model, le = train_sector_aware_model()
    
    # Re-evaluate tech companies
    results = evaluate_tech_companies_with_calibration()
