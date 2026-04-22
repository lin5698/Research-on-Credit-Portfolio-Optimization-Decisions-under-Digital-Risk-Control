"""
Stage 3: Portfolio Optimization Module (SA-NA Algorithm)
"""
import random
import numpy as np

class LoanPortfolioOptimizer:
    def __init__(self, companies_df, ri_scores, pd_values, config):
        self.df = companies_df
        self.ri = ri_scores
        self.pd = pd_values
        self.cfg = config
        self.n = len(companies_df)
        self.pareto_front = [] # Stores non-dominated solutions [(solution, objectives), ...]

    def get_churn_rate(self, rate):
        """
        Calculate Customer Churn Rate based on Interest Rate (Figure 5).
        Assumes a Sigmoid relationship: Higher rate -> Higher churn (probability of rejection).
        
        Churn(r) = 1 / (1 + exp(-k * (r - r0)))
        Center (r0) at 12% (0.12), Steepness (k) = 50.
        """
        # Vectorized implementation if rate is array
        k = 50
        r0 = 0.12
        return 1 / (1 + np.exp(-k * (rate - r0)))

    def objective_functions(self, solution):
        """
        Calculate Dual Objectives:
        1. Max RAROC (Risk-Adjusted Return on Capital) -> Converted to Min (-RAROC)
        2. Min CVaR (Conditional Value at Risk)
        
        solution: List of tuples [(amount, rate, decision_bool), ...]
        """
        amounts = np.array([s[0] for s in solution])
        rates = np.array([s[1] for s in solution])
        decisions = np.array([s[2] for s in solution])
      
        active_idx = decisions == 1
        if not np.any(active_idx):
            return 9999, 9999 # Penalty for empty solution
          
        # --- Calculate RAROC ---
        # --- Calculate RAROC ---
        # RAROC = (Income - EL) / EC
        # Income = Sum(Amount * Rate * (1 - PD)) - Cost of Funds (assumed 0 or included in net rate)
        # EL (Expected Loss) = Sum(Amount * PD * LGD). LGD=1 assumed.
        # EC (Economic Capital) = UL (Unexpected Loss) = Sum(Amount * sqrt(PD*(1-PD)))
        
        # Income component: Amount * Rate * (1 - PD)
        # Note: The user formula is Sum(Li * ri * (1-pi)) - EL / EC
        # This implies the numerator is (Expected Income - Expected Loss).
        # Let's follow the formula strictly:
        # Term 1: Sum(Li * ri * (1-pi)) -> Expected Interest Income
        # Term 2: EL = Sum(Li * pi) -> Expected Principal Loss (assuming LGD=1)
        
        # --- Calculate Churn Rate ---
        # The effective loan amount is reduced by the churn rate (probability of customer rejecting)
        # Expected Amount = Amount * (1 - Churn)
        churn_rates = self.get_churn_rate(rates[active_idx])
        acceptance_prob = 1 - churn_rates
        
        # Effective amounts for calculation (Expected Exposure)
        eff_amounts = amounts[active_idx] * acceptance_prob
        
        # Income component: Effective Amount * Rate * (1 - PD)
        # Note: The user formula is Sum(Li * ri * (1-pi)) - EL / EC
        # We apply acceptance probability to the whole deal structure
        
        income = np.sum(eff_amounts * rates[active_idx] * (1 - self.pd[active_idx]))
        el = np.sum(eff_amounts * self.pd[active_idx])
        
        # Economic Capital (EC) simplified as Unexpected Loss (UL)
        ul = np.sum(eff_amounts * np.sqrt(self.pd[active_idx] * (1 - self.pd[active_idx])))
        
        # RAROC = (Income - EL) / EC
        # Note: If Income < EL, RAROC is negative.
        raroc = (income - el) / (ul + 1e-6)
      
        # --- Calculate CVaR (Monte Carlo Simulation) ---
        n_sims = 1000
        # Vectorized simulation
        # Create a matrix of random draws: (n_sims, n_active_loans)
        n_active = np.sum(active_idx)
        rand_draws = np.random.rand(n_sims, n_active)
        
        # PDs for active loans
        active_pds = self.pd[active_idx]
        active_amounts = amounts[active_idx]
        
        # Check defaults: random < PD
        defaults = rand_draws < active_pds
        
        # Calculate loss for each simulation: Sum(Amount * Default)
        # We assume LGD = 100% for simplification as per code skeleton
        # Calculate loss for each simulation: Sum(Amount * Default)
        # We assume LGD = 100% for simplification as per code skeleton
        # For CVaR, we should also consider that some loans didn't happen due to churn?
        # Simulation approach:
        # 1. Determine if loan is accepted (Bernoulli(1-Churn))
        # 2. If accepted, Determine if default (Bernoulli(PD))
        # However, for expected CVaR, using effective amount is a valid approximation for portfolio risk
        # or we can simulate acceptance too. Let's simulate acceptance for robustness.
        
        # Simulate Acceptance
        # random draws for acceptance: (n_sims, n_active)
        rand_accept = np.random.rand(n_sims, n_active)
        accepted = rand_accept < acceptance_prob # Broadcast acceptance_prob
        
        # Realized amounts in simulation
        sim_amounts = amounts[active_idx] * accepted
        
        # Calculate losses on accepted loans
        # defaults is (n_sims, n_active)
        # losses is sum over companies
        losses = np.sum(sim_amounts * defaults, axis=1)
          
        # Calculate Tail Loss
        var_idx = int(n_sims * self.cfg.CONFIDENCE_LEVEL)
        sorted_losses = np.sort(losses)
        # CVaR is the mean of losses beyond VaR
        if var_idx < n_sims:
            cvar = np.mean(sorted_losses[var_idx:])
        else:
            cvar = sorted_losses[-1]
      
        return -raroc, cvar # Return negative RAROC for minimization

    def check_constraints(self, solution):
        amounts = np.array([s[0] for s in solution])
        decisions = np.array([s[2] for s in solution])

        # Reject degenerate portfolios with no approved loans.
        if np.sum(decisions) == 0:
            return False
      
        total_loan = np.sum(amounts * decisions)
        if total_loan > self.cfg.TOTAL_BUDGET:
            return False
        return True

    def generate_neighbor(self, solution, perturbation_scale=0.1):
        """Generate Neighbor Solution (NA part)"""
        new_sol = [list(s) for s in solution]
        idx = random.randint(0, self.n - 1)
      
        # Random perturbation: change amount, rate, or decision
        choice = random.random()
        if choice < 0.4: # Change Amount
            new_sol[idx][0] *= (1 + random.uniform(-perturbation_scale, perturbation_scale))
            new_sol[idx][0] = min(max(new_sol[idx][0], 0), self.cfg.MAX_SINGLE_LOAN)
        elif choice < 0.7: # Change Rate
            # Rate usually has bounds too, e.g., 0.03 to 0.15
            new_sol[idx][1] *= (1 + random.uniform(-perturbation_scale, perturbation_scale))
            new_sol[idx][1] = min(max(new_sol[idx][1], 0.01), 0.20) # Reasonable bounds
        else: # Flip Decision
            new_sol[idx][2] = 1 - new_sol[idx][2]
          
        return [tuple(s) for s in new_sol]

    def update_pareto(self, solution, objs):
        """Update Pareto Front"""
        is_dominated = False
        to_remove = []
      
        for p_sol, p_objs in self.pareto_front:
            # If current solution is dominated by an existing one
            # (Existing has smaller/equal -RAROC AND smaller/equal CVaR, and at least one is strictly smaller)
            if p_objs[0] <= objs[0] and p_objs[1] <= objs[1] and (p_objs[0] < objs[0] or p_objs[1] < objs[1]):
                is_dominated = True
                break
            # If current solution dominates an existing one
            if objs[0] <= p_objs[0] and objs[1] <= p_objs[1] and (objs[0] < p_objs[0] or objs[1] < p_objs[1]):
                to_remove.append((p_sol, p_objs))
      
        if not is_dominated:
            for item in to_remove:
                self.pareto_front.remove(item)
            # Avoid duplicates
            if not any(np.allclose(objs, p_objs) for _, p_objs in self.pareto_front):
                self.pareto_front.append((solution, objs))
                return True # Found new non-dominated solution
        return False

    def run_sa_na(self):
        print("Stage 3: Running SA-NA Optimization...")
        # Initialize Solution with heuristic: prioritize low-risk companies
        # Sort by RI (lower is better) and PD (lower is better)
        risk_scores = self.ri + self.pd * 100  # Combined risk metric
        sorted_indices = np.argsort(risk_scores)
        
        # Initialize: approve top 50% low-risk companies with reasonable amounts
        current_sol = []
        for i in range(self.n):
            if i in sorted_indices[:self.n//2]:
                # Low risk: approve with moderate amount
                amount = random.uniform(self.cfg.MAX_SINGLE_LOAN * 0.3, self.cfg.MAX_SINGLE_LOAN * 0.7)
                rate = random.uniform(0.04, 0.08)  # Lower rates for low risk
                decision = 1
            else:
                # High risk: initially reject or small amount
                amount = random.uniform(1e5, self.cfg.MAX_SINGLE_LOAN * 0.3)
                rate = random.uniform(0.08, 0.15)  # Higher rates for high risk
                decision = 0
            current_sol.append((amount, rate, decision))
        
        current_objs = self.objective_functions(current_sol)
        
        # Add initial solution to Pareto if valid
        if self.check_constraints(current_sol):
            self.update_pareto(current_sol, current_objs)
      
        temp = self.cfg.SA_TEMP_INIT
      
        for i in range(self.cfg.SA_ITERATIONS):
            if i % 50 == 0:
                print(f"Iteration {i}/{self.cfg.SA_ITERATIONS}, Temp: {temp:.2f}, Pareto Size: {len(self.pareto_front)}")
                
            # SA Global Search
            new_sol = self.generate_neighbor(current_sol)
            if not self.check_constraints(new_sol):
                continue
              
            new_objs = self.objective_functions(new_sol)
          
            # Metropolis Criterion (based on weighted sum for acceptance probability)
            # We normalize objectives roughly to make them comparable for the delta calculation
            # RAROC is around 0.1-1.0, CVaR is large (millions). 
            # This is a heuristic. Let's just use the raw sum if we assume they are scaled or just use a simple logic.
            # Better approach for Multi-objective SA: Accept if it dominates current, or with prob if dominated.
            # Here we stick to the provided logic: delta based on sum.
            # Note: new_objs[0] is -RAROC (small), new_objs[1] is CVaR (large). 
            # Direct sum is dominated by CVaR. This might bias towards minimizing CVaR only.
            # Let's keep it as is for fidelity to the skeleton, but acknowledge the scale issue.
            
            # Metropolis Criterion
            # Normalize objectives to handle scale difference
            # RAROC is approx 0.1-1.0. CVaR is approx 1e5-1e6.
            # We scale CVaR by 1e-6 to make it comparable to RAROC.
            # Objective: Maximize RAROC (Minimize -RAROC), Minimize CVaR.
            # Current objs are [-RAROC, CVaR].
            # We want to minimize the weighted sum.
            
            w_raroc = 1.0
            w_cvar = 1e-6
            
            current_energy = current_objs[0] * w_raroc + current_objs[1] * w_cvar
            new_energy = new_objs[0] * w_raroc + new_objs[1] * w_cvar
            
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < np.exp(-delta / temp):
                current_sol = new_sol
                current_objs = new_objs
              
                # Update Pareto Front
                is_new_pareto = self.update_pareto(current_sol, current_objs)
              
                # NA Local Exploitation (if new Pareto solution found)
                if is_new_pareto:
                    # Search intensely around this solution
                    for _ in range(self.cfg.NA_NEIGHBOR_SIZE):
                        na_sol = self.generate_neighbor(current_sol, perturbation_scale=0.05) # Small step
                        if self.check_constraints(na_sol):
                            na_objs = self.objective_functions(na_sol)
                            self.update_pareto(na_sol, na_objs)
          
            temp *= self.cfg.SA_COOLING_RATE
          
        return self.pareto_front
