"""
Stage 2: Risk Quantification Module (EWM + TOPSIS)
"""
import numpy as np
import pandas as pd

class RiskQuantifier:
    def __init__(self, config):
        self.cfg = config

    def entropy_weight(self, data):
        """Calculate Entropy Weights"""
        # Normalize (Min-Max Scaling)
        # Add small epsilon to avoid division by zero
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        range_val = data_max - data_min
        range_val[range_val == 0] = 1e-6 # Avoid division by zero
        
        normalized_data = (data - data_min) / range_val
        
        n, m = normalized_data.shape
        if n <= 1:
            return np.ones(m) / m
            
        k = 1.0 / np.log(n)
        
        # Calculate probability matrix
        # Add epsilon to avoid log(0)
        p = normalized_data / (normalized_data.sum(axis=0) + 1e-6)
        
        # Calculate entropy
        e = -k * (p * np.log(p + 1e-6)).sum(axis=0)
        
        # Calculate redundancy
        d = 1 - e
        
        # Calculate weights
        if d.sum() == 0:
            w = np.ones(m) / m
        else:
            w = d / d.sum()
            
        return w

    def topsis(self, data, weights, is_benefit):
        """TOPSIS Calculation"""
        # Weighted matrix
        z = data * weights
      
        # Ideal solutions
        z_plus = np.zeros(data.shape[1])
        z_minus = np.zeros(data.shape[1])
      
        # Correct for cost indicators
        # Return Rate (Debt/Equity) is cost type, others are benefit type
        for i, benefit in enumerate(is_benefit):
            if benefit:
                z_plus[i] = np.max(z[:, i])
                z_minus[i] = np.min(z[:, i])
            else:
                z_plus[i] = np.min(z[:, i])
                z_minus[i] = np.max(z[:, i])
      
        # Euclidean distances
        d_plus = np.sqrt(((z - z_plus) ** 2).sum(axis=1))
        d_minus = np.sqrt(((z - z_minus) ** 2).sum(axis=1))
      
        # Relative closeness
        score = d_minus / (d_plus + d_minus + 1e-6)
        return score

    def calculate_risk_index(self, df, pd_values):
        print("Stage 2: Calculating Risk Index (RI)...")
      
        # Extract 5 core indicators
        cols = list(self.cfg.FEATURE_MAP.values())
        # Ensure columns exist
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for RI calculation: {missing}")
            
        data_matrix = df[cols].values
      
        # 1. First Evaluation: Corporate Hard Power Score
        # Assumptions: First 4 are Benefit (True), Last 1 (Debt/Equity) is Cost (False)
        is_benefit = [True, True, True, True, False]
      
        w1 = self.entropy_weight(data_matrix)
        strength_score = self.topsis(data_matrix, w1, is_benefit)
      
        # 2. Second Evaluation: Combine Strength Score and Credit Rating
        # The paper mentions using "Linear Weighting Method" for the second level
        # Input 1: Strength Score (from Level 1)
        # Input 2: Credit Rating Score. The paper says PD is mapped to A-D then numericalized.
        # Here we use (1 - PD) as a high-precision numerical representation of Credit Rating (Higher is better).
        
        # Normalize inputs for Level 2 EWM
        # Strength score is already 0-1 from TOPSIS.
        # (1-PD) is 0-1.
        rating_score = 1 - pd_values
        
        secondary_matrix = np.column_stack((strength_score, rating_score))
        
        # Calculate weights for Level 2 using EWM
        w2 = self.entropy_weight(secondary_matrix)
        print(f"Level 2 Weights - Strength: {w2[0]:.4f}, Rating: {w2[1]:.4f}")
        
        # Comprehensive Credit Ability (Linear Weighting)
        # Score = w1 * Strength + w2 * Rating
        comprehensive_score = (secondary_matrix * w2).sum(axis=1)
        
        # Convert to Risk Index RI (0-100)
        # RI is negatively correlated with Credit Ability
        # We normalize comprehensive_score to 0-1 first if it's not (it should be if weights sum to 1 and inputs are 0-1)
        # Then RI = (1 - score) * 100
        
        # Ensure score is 0-1
        max_score = comprehensive_score.max()
        min_score = comprehensive_score.min()
        if max_score > min_score:
            norm_score = (comprehensive_score - min_score) / (max_score - min_score)
        else:
            norm_score = comprehensive_score
            
        ri_scores = (1 - norm_score) * 100
        return ri_scores
