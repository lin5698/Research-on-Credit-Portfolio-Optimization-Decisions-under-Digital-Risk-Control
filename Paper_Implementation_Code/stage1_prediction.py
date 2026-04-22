"""
Stage 1: Prediction Module (XGBoost + SHAP)
"""
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class RiskPredictor:
    def __init__(self):
        # Optimized hyperparameters from advanced randomized search
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=3,
            subsample=1.0,
            colsample_bytree=0.6,
            gamma=0.3,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            use_label_encoder=False,
            # scale_pos_weight helps with class imbalance (Negative/Positive)
            # We will set this dynamically during training or use a fixed heuristic
            # For multiclass, XGBoost doesn't use scale_pos_weight directly in the same way as binary
            # But we can use sample_weight in fit().
            # However, if we treat this as binary (Default vs Non-Default) for the paper's context:
            # The paper mentions scale_pos_weight, implying a binary or one-vs-rest approach.
            # Our current setup is 'multi:softprob'. 
            # To align with paper, let's keep multi but add weight handling in fit.
            eval_metric='mlogloss'
        )
        self.calibrated_model = None

    def train(self, X, y):
        print("Stage 1: Training XGBoost Model...")
        
        # Calculate class weights for imbalance handling
        # Simple heuristic: max_class_count / class_counts
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight(class_weight='balanced', y=y)
        
        self.model.fit(X, y, sample_weight=sample_weights)
        
        # Temperature Scaling (Simplified implementation)
        # We will fit a simple Logistic Regression on the logits of the validation set
        # or just use CalibratedClassifierCV from sklearn which is robust.
        # The paper mentions "Temperature Scaling", which is often for Neural Nets, 
        # but Isotonic/Sigmoid calibration is standard for trees.
        # Let's use CalibratedClassifierCV with 'sigmoid' (Platt Scaling) which is close to Temperature Scaling concept for binary.
        # For multiclass, it works per class.
        from sklearn.calibration import CalibratedClassifierCV
        print("Applying Probability Calibration...")
        self.calibrated_model = CalibratedClassifierCV(self.model, method='sigmoid', cv='prefit')
        self.calibrated_model.fit(X, y)
        
        return self.calibrated_model

    def predict_pd(self, X):
        """
        Predict Probability of Default (PD).
        Assumes the highest class index corresponds to the highest risk (Default).
        """
        # Returns probability of default (assuming class 3 is default/high risk)
        if self.calibrated_model:
            probs = self.calibrated_model.predict_proba(X)
        else:
            probs = self.model.predict_proba(X)
        # According to paper, PD might be sum of speculative grades or just the worst grade
        # Here we take the probability of the highest risk class (last column)
        pd_values = probs[:, -1] 
        return pd_values

    def explain_model(self, X):
        print("Stage 1: Generating SHAP Explanations...")
        try:
            # Use TreeExplainer with the underlying booster to avoid wrapper issues
            # This fixes the 'base_score' parsing error common with newer XGBoost/SHAP versions
            # Also ensure feature names are passed if possible, but booster might lose them.
            # We can pass model directly if we accept the warning, or use the booster.
            # Let's try passing the model first, but if it fails, fallback to booster.
            try:
                explainer = shap.TreeExplainer(self.model)
            except Exception:
                explainer = shap.TreeExplainer(self.model.get_booster())
                
            shap_values = explainer.shap_values(X)
            
            # shap_values from Explainer is an Explanation object
            # We need to extract values for plotting
            if len(shap_values.shape) == 3:
                # (samples, features, classes)
                # Take the last class (High Risk)
                shap_values_high_risk = shap_values[..., -1]
            else:
                shap_values_high_risk = shap_values

            # Plot Global Feature Importance (Paper Fig 7)
            plt.figure()
            shap.summary_plot(shap_values_high_risk, X, plot_type="bar", show=False)
            plt.title("Global Feature Importance (High Risk Class)")
            plt.tight_layout()
            plt.savefig('shap_global_importance.png')
            print("Saved SHAP plot to shap_global_importance.png")
            
            return shap_values
            
        except Exception as e:
            print(f"Warning: TreeExplainer failed ({e}). Falling back to KernelExplainer...")
            try:
                # Fallback to KernelExplainer (slower but model-agnostic)
                # Use a small background dataset for speed (e.g., K-means summary)
                # X might be a DataFrame, convert to numpy for kmeans
                X_summary = shap.kmeans(X, 10)
                explainer = shap.KernelExplainer(self.model.predict_proba, X_summary)
                shap_values = explainer.shap_values(X)
                
                # KernelExplainer returns a list for multiclass
                if isinstance(shap_values, list):
                    shap_values_high_risk = shap_values[-1]
                else:
                    shap_values_high_risk = shap_values

                plt.figure()
                shap.summary_plot(shap_values_high_risk, X, plot_type="bar", show=False)
                plt.title("Global Feature Importance (High Risk Class)")
                plt.tight_layout()
                plt.savefig('shap_global_importance.png')
                print("Saved SHAP plot to shap_global_importance.png")
                
                # 2. Individual Decision Attribution (Force Plot)
                # Plot for the first sample in the set (or a specific high risk one)
                try:
                    # Find a high risk sample index
                    high_risk_idx = np.argmax(shap_values_high_risk.sum(axis=1))
                    print(f"Generating Force Plot for sample index {high_risk_idx}...")
                    
                    plt.figure()
                    shap.plots.force(explainer.expected_value[-1], shap_values_high_risk[high_risk_idx], X.iloc[high_risk_idx], matplotlib=True, show=False)
                    plt.savefig('shap_force_plot_sample.png')
                    print("Saved SHAP force plot to shap_force_plot_sample.png")
                except Exception as e_force:
                    print(f"Could not generate force plot: {e_force}")

                # 3. Feature Interaction (Dependence Plot)
                # Plot interaction between top 2 features
                try:
                    # Get top 2 features by mean absolute SHAP value
                    mean_shap = np.abs(shap_values_high_risk).mean(axis=0)
                    top_indices = np.argsort(mean_shap)[-2:]
                    top_feats = X.columns[top_indices]
                    print(f"Generating Dependence Plot for {top_feats[1]} vs {top_feats[0]}...")
                    
                    plt.figure()
                    shap.dependence_plot(top_feats[1], shap_values_high_risk, X, interaction_index=top_feats[0], show=False)
                    plt.tight_layout()
                    plt.savefig('shap_dependence_plot.png')
                    print("Saved SHAP dependence plot to shap_dependence_plot.png")
                except Exception as e_dep:
                    print(f"Could not generate dependence plot: {e_dep}")

                return shap_values
                
            except Exception as e2:
                print(f"Warning: SHAP explanation failed completely: {e2}")
                return None
