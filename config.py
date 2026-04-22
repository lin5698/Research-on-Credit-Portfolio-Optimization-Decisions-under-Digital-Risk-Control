"""
Global Configuration Module
"""

class Config:
    # Data Paths
    DATA_PATH = 'data/corporate_credit_rating.csv'
  
    # Feature Mapping (Paper Table 1 -> Dataset Columns)
    FEATURE_MAP = {
        'Activity_Level': 'currentRatio',          # 活力程度
        'Sales_Capability': 'netProfitMargin',    # 销售能力
        'Cooperation_Capability': 'assetTurnover', # 合作能力
        'Marketing_Capability': 'operatingCashFlowSalesRatio',        # 营销能力
        'Return_Rate': 'debtEquityRatio'          # 被退货率(代理变量：潜在风险率)
    }
  
    # Rating Mapping
    RATING_MAP = {
        'AAA': 0, 'AA': 0, 'A': 0,  # A Grade (Low Risk)
        'BBB': 1,                   # B Grade
        'BB': 2,                    # C Grade
        'B': 3, 'CCC': 3, 'CC': 3, 'C': 3, 'D': 3 # D Grade (High Risk)
    }
  
    # Optimization Constraints
    TOTAL_BUDGET = 100_000_000  # Total Budget (e.g., 100 Million)
    RISK_FREE_RATE = 0.03       # Risk-free Rate
    CONFIDENCE_LEVEL = 0.95     # CVaR Confidence Level
    MAX_SINGLE_LOAN = 10_000_000 # Max Loan per Company
  
    # SA-NA Algorithm Parameters
    SA_TEMP_INIT = 1000         # Initial Temperature (Optimized)
    SA_COOLING_RATE = 0.95      # Cooling Rate (Optimized)
    SA_ITERATIONS = 500         # Outer Iterations
    NA_NEIGHBOR_SIZE = 20       # Neighborhood Search Size
