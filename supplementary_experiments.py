from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions
from scipy.stats import mannwhitneyu
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LinearRegression
from sklearn.utils.class_weight import compute_sample_weight

from config import Config
from data_loader import DataLoader
from stage2_evaluation import RiskQuantifier
from stage3_optimization import LoanPortfolioOptimizer


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "output" / "experiments" / "reviewer_supplement"

BEST_XGB_PARAMS = {
    "subsample": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
    "n_estimators": 500,
    "min_child_weight": 3,
    "max_depth": 4,
    "learning_rate": 0.03,
    "gamma": 0.3,
    "colsample_bytree": 0.6,
}

MATCHED_EVAL_BUDGET = 800


def non_dominated_mask(F: np.ndarray) -> np.ndarray:
    n = len(F)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominates_i = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if np.any(dominates_i):
            keep[i] = False
            continue
        dominated_by_i = np.all(F[i] <= F, axis=1) & np.any(F[i] < F, axis=1)
        keep[dominated_by_i] = False
        keep[i] = True
    return keep


@dataclass
class RunResult:
    algorithm: str
    seed: int
    runtime_s: float
    objective_calls: int
    pareto_size: int
    best_raroc: float
    best_cvar: float
    hv: float | None = None
    igd: float | None = None
    na_triggers: int | None = None
    na_evals: int | None = None
    accepted_moves: int | None = None
    front: np.ndarray | None = None


class DeterministicPortfolioEvaluator:
    def __init__(
        self,
        pd_values: np.ndarray,
        cfg: Config,
        scenarios: np.ndarray,
    ) -> None:
        self.pd = np.asarray(pd_values, dtype=float)
        self.cfg = cfg
        self.scenarios = np.asarray(scenarios, dtype=float)
        self.calls = 0

    def evaluate(
        self,
        amounts: np.ndarray,
        rates: np.ndarray,
        decisions: np.ndarray,
        allow_penalty: bool = False,
    ) -> tuple[float, float]:
        self.calls += 1

        amounts = np.asarray(amounts, dtype=float)
        rates = np.asarray(rates, dtype=float)
        decisions = np.asarray(decisions, dtype=int)
        active = decisions == 1

        if not np.any(active):
            return (9999.0, 9999.0)

        total_loan = float(np.sum(amounts[active]))
        budget_excess = max(0.0, total_loan - self.cfg.TOTAL_BUDGET)

        income = np.sum(amounts[active] * rates[active] * (1.0 - self.pd[active]))
        el = np.sum(amounts[active] * self.pd[active])
        ec = np.sum(amounts[active] * np.sqrt(self.pd[active] * (1.0 - self.pd[active])))
        raroc = float((income - el) / (ec + 1e-6))

        losses = (self.scenarios[:, active] < self.pd[active]).dot(amounts[active])
        sorted_losses = np.sort(losses)
        var_idx = int(len(sorted_losses) * self.cfg.CONFIDENCE_LEVEL)
        cvar = float(np.mean(sorted_losses[var_idx:])) if var_idx < len(sorted_losses) else float(sorted_losses[-1])

        if allow_penalty and budget_excess > 0:
            penalty_scale = budget_excess / self.cfg.TOTAL_BUDGET
            return (-raroc + 100.0 * penalty_scale, cvar + 1e8 * penalty_scale)

        if budget_excess > 0:
            return (9999.0, 9999.0)

        return (-raroc, cvar)


class SANAExperimentOptimizer(LoanPortfolioOptimizer):
    def __init__(
        self,
        companies_df: pd.DataFrame,
        ri_scores: np.ndarray,
        pd_values: np.ndarray,
        config: Config,
        evaluator: DeterministicPortfolioEvaluator,
        enable_na: bool,
        eval_budget: int,
    ) -> None:
        super().__init__(companies_df, ri_scores, pd_values, config)
        self.evaluator = evaluator
        self.enable_na = enable_na
        self.eval_budget = int(eval_budget)
        self.na_triggers = 0
        self.na_evals = 0
        self.accepted_moves = 0

    def objective_functions(self, solution):
        amounts = np.array([s[0] for s in solution], dtype=float)
        rates = np.array([s[1] for s in solution], dtype=float)
        decisions = np.array([s[2] for s in solution], dtype=int)
        return self.evaluator.evaluate(amounts, rates, decisions, allow_penalty=False)

    def run(self) -> list[tuple[list[tuple[float, float, int]], tuple[float, float]]]:
        risk_scores = self.ri + self.pd * 100.0
        sorted_indices = np.argsort(risk_scores)
        current_sol = []
        for i in range(self.n):
            if i in sorted_indices[: self.n // 2]:
                amount = random.uniform(self.cfg.MAX_SINGLE_LOAN * 0.3, self.cfg.MAX_SINGLE_LOAN * 0.7)
                rate = random.uniform(0.04, 0.08)
                decision = 1
            else:
                amount = random.uniform(1e5, self.cfg.MAX_SINGLE_LOAN * 0.3)
                rate = random.uniform(0.08, 0.15)
                decision = 0
            current_sol.append((amount, rate, decision))

        current_objs = self.objective_functions(current_sol)
        if self.check_constraints(current_sol):
            self.update_pareto(current_sol, current_objs)

        temp = self.cfg.SA_TEMP_INIT
        max_iterations = max(self.cfg.SA_ITERATIONS, self.eval_budget * 20)
        iterations = 0

        while self.evaluator.calls < self.eval_budget and iterations < max_iterations:
            iterations += 1
            new_sol = self.generate_neighbor(current_sol)
            if not self.check_constraints(new_sol):
                temp *= self.cfg.SA_COOLING_RATE
                continue

            if self.evaluator.calls >= self.eval_budget:
                break
            new_objs = self.objective_functions(new_sol)
            w_raroc = 1.0
            w_cvar = 1e-6
            current_energy = current_objs[0] * w_raroc + current_objs[1] * w_cvar
            new_energy = new_objs[0] * w_raroc + new_objs[1] * w_cvar
            delta = new_energy - current_energy

            if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-12)):
                self.accepted_moves += 1
                current_sol = new_sol
                current_objs = new_objs
                is_new_pareto = self.update_pareto(current_sol, current_objs)

                if self.enable_na and is_new_pareto:
                    self.na_triggers += 1
                    for _ in range(self.cfg.NA_NEIGHBOR_SIZE):
                        if self.evaluator.calls >= self.eval_budget:
                            break
                        na_sol = self.generate_neighbor(current_sol, perturbation_scale=0.05)
                        if self.check_constraints(na_sol):
                            na_objs = self.objective_functions(na_sol)
                            self.na_evals += 1
                            self.update_pareto(na_sol, na_objs)

            temp *= self.cfg.SA_COOLING_RATE

        return self.pareto_front


class CreditPortfolioProblem(ElementwiseProblem):
    def __init__(self, n_companies: int, cfg: Config, evaluator: DeterministicPortfolioEvaluator):
        self.n_companies = n_companies
        self.cfg = cfg
        self.eval = evaluator

        xl = np.zeros(3 * n_companies, dtype=float)
        xu = np.zeros(3 * n_companies, dtype=float)
        for i in range(n_companies):
            xl[3 * i] = 0.0
            xu[3 * i] = cfg.MAX_SINGLE_LOAN
            xl[3 * i + 1] = 0.01
            xu[3 * i + 1] = 0.20
            xl[3 * i + 2] = 0.0
            xu[3 * i + 2] = 1.0

        super().__init__(n_var=3 * n_companies, n_obj=2, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        amounts = np.array([x[3 * i] for i in range(self.n_companies)], dtype=float)
        rates = np.array([x[3 * i + 1] for i in range(self.n_companies)], dtype=float)
        decisions = np.array([1 if x[3 * i + 2] >= 0.5 else 0 for i in range(self.n_companies)], dtype=int)
        out["F"] = np.array(self.eval.evaluate(amounts, rates, decisions, allow_penalty=True))


def parse_best_params() -> dict:
    base = {
        "objective": "multi:softprob",
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
    }
    base.update(BEST_XGB_PARAMS)
    return base


def load_pipeline():
    cfg = Config()
    loader = DataLoader(cfg)
    df_full = loader.load_and_preprocess()
    X_train, y_train, df_opt = loader.split_data(df_full)

    model = xgb.XGBClassifier(**parse_best_params())
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(X_train, y_train)

    feature_cols = list(cfg.FEATURE_MAP.values())
    X_opt = df_opt[feature_cols].copy()
    if "sector_encoded" in df_opt.columns:
        X_opt["sector"] = df_opt["sector_encoded"]

    pd_values = calibrated.predict_proba(X_opt)[:, -1]
    ri_scores = RiskQuantifier(cfg).calculate_risk_index(df_opt, pd_values)
    return cfg, df_full, df_opt, pd_values, ri_scores


def multicollinearity_report(df_full: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = list(cfg.FEATURE_MAP.values())
    X = df_full[cols].copy().replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    corr = X.corr().round(3)

    vif_rows = []
    for col in X.columns:
        y = X[col].values
        X_other = X.drop(columns=[col]).values
        r2 = LinearRegression().fit(X_other, y).score(X_other, y)
        vif = math.inf if r2 >= 0.999999 else 1.0 / (1.0 - r2)
        vif_rows.append({"variable": col, "r2": round(float(r2), 4), "vif": round(float(vif), 3)})

    return corr, pd.DataFrame(vif_rows)


def make_scenarios(seed: int, n_sims: int, n_companies: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n_sims, n_companies))


def front_to_array(front) -> np.ndarray:
    if front is None:
        return np.empty((0, 2))
    if isinstance(front, np.ndarray):
        return front
    arr = np.array([list(objs) for _, objs in front], dtype=float)
    if arr.size == 0:
        return np.empty((0, 2))
    mask = np.isfinite(arr).all(axis=1)
    return arr[mask]


def summarize_front(arr: np.ndarray) -> tuple[int, float, float]:
    if len(arr) == 0:
        return 0, float("nan"), float("nan")
    return len(arr), float(-np.min(arr[:, 0])), float(np.min(arr[:, 1]))


def run_sana_series(
    name: str,
    seeds: list[int],
    budget: int,
    cfg: Config,
    df_opt: pd.DataFrame,
    ri_scores: np.ndarray,
    pd_values: np.ndarray,
    enable_na: bool,
) -> list[RunResult]:
    results = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        scenarios = make_scenarios(seed, 1000, len(df_opt))
        evaluator = DeterministicPortfolioEvaluator(pd_values, cfg, scenarios)
        optimizer = SANAExperimentOptimizer(
            df_opt, ri_scores, pd_values, cfg, evaluator, enable_na=enable_na, eval_budget=budget
        )
        t0 = time.time()
        front = optimizer.run()
        runtime = time.time() - t0
        arr = front_to_array(front)
        pareto_size, best_raroc, best_cvar = summarize_front(arr)
        results.append(
            RunResult(
                algorithm=name,
                seed=seed,
                runtime_s=runtime,
                objective_calls=evaluator.calls,
                pareto_size=pareto_size,
                best_raroc=best_raroc,
                best_cvar=best_cvar,
                na_triggers=optimizer.na_triggers,
                na_evals=optimizer.na_evals,
                accepted_moves=optimizer.accepted_moves,
                front=arr,
            )
        )
    return results
def get_algorithm(name: str, pop_size: int):
    if name == "NSGA-II":
        return NSGA2(pop_size=pop_size)
    if name == "SPEA2":
        return SPEA2(pop_size=pop_size)
    if name == "MOEA/D":
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=pop_size - 1)
        return MOEAD(ref_dirs=ref_dirs, n_neighbors=20, prob_neighbor_mating=0.9)
    raise ValueError(name)


def run_pymoo_series(
    name: str,
    seeds: list[int],
    budget: int,
    cfg: Config,
    df_opt: pd.DataFrame,
    pd_values: np.ndarray,
) -> list[RunResult]:
    results = []
    pop_size = 100
    for seed in seeds:
        scenarios = make_scenarios(seed, 1000, len(df_opt))
        evaluator = DeterministicPortfolioEvaluator(pd_values, cfg, scenarios)
        problem = CreditPortfolioProblem(len(df_opt), cfg, evaluator)
        algorithm = get_algorithm(name, pop_size)
        termination = get_termination("n_eval", budget)

        t0 = time.time()
        res = minimize(problem, algorithm, termination, seed=seed, verbose=False)
        runtime = time.time() - t0
        arr = np.asarray(res.F, dtype=float) if res.F is not None else np.empty((0, 2))
        if arr.size:
            arr = arr[np.isfinite(arr).all(axis=1)]
            if len(arr):
                arr = arr[non_dominated_mask(arr)]
        pareto_size, best_raroc, best_cvar = summarize_front(arr)
        results.append(
            RunResult(
                algorithm=name,
                seed=seed,
                runtime_s=runtime,
                objective_calls=evaluator.calls,
                pareto_size=pareto_size,
                best_raroc=best_raroc,
                best_cvar=best_cvar,
                front=arr,
            )
        )
    return results


def assign_front_metrics(results: list[RunResult]) -> None:
    fronts = [r.front for r in results if r.front is not None and len(r.front)]
    all_F = np.vstack(fronts)
    nd = all_F[non_dominated_mask(all_F)]
    f_min = all_F.min(axis=0)
    f_max = all_F.max(axis=0)
    scale = np.where(np.abs(f_max - f_min) < 1e-12, 1.0, f_max - f_min)

    ref_front = (nd - f_min) / scale
    ref_point = np.array([1.1, 1.1])
    hv_indicator = HV(ref_point=ref_point)
    igd_indicator = IGD(ref_front)

    for r in results:
        if r.front is None or not len(r.front):
            r.hv = float("nan")
            r.igd = float("nan")
            continue
        norm = (r.front - f_min) / scale
        norm = np.clip(norm, 0.0, 1.1)
        r.hv = float(hv_indicator(norm))
        r.igd = float(igd_indicator(norm))


def results_frame(results: list[RunResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "algorithm": r.algorithm,
                "seed": r.seed,
                "runtime_s": r.runtime_s,
                "objective_calls": r.objective_calls,
                "pareto_size": r.pareto_size,
                "best_raroc": r.best_raroc,
                "best_cvar": r.best_cvar,
                "hv": r.hv,
                "igd": r.igd,
                "na_triggers": r.na_triggers,
                "na_evals": r.na_evals,
                "accepted_moves": r.accepted_moves,
            }
        )
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["runtime_s", "objective_calls", "pareto_size", "best_raroc", "best_cvar", "hv", "igd"]
    pieces = []
    grouped = df.groupby("algorithm", dropna=False)
    for metric in metrics:
        stats = grouped[metric].agg(["mean", "std"]).reset_index()
        stats["metric"] = metric
        pieces.append(stats)
    summary = pd.concat(pieces, ignore_index=True)
    return summary[["algorithm", "metric", "mean", "std"]]


def pairwise_tests(df: pd.DataFrame, baseline_algorithms: list[str]) -> pd.DataFrame:
    rows = []
    sana = df[df["algorithm"] == "SA-NA"]
    for alg in baseline_algorithms:
        comp = df[df["algorithm"] == alg]
        for metric, alternative in [("hv", "greater"), ("igd", "less")]:
            x = sana[metric].dropna()
            y = comp[metric].dropna()
            if len(x) and len(y):
                stat = mannwhitneyu(x, y, alternative=alternative)
                rows.append(
                    {
                        "comparison": f"SA-NA vs {alg}",
                        "metric": metric,
                        "alternative": alternative,
                        "u_statistic": float(stat.statistic),
                        "p_value": float(stat.pvalue),
                    }
                )
    return pd.DataFrame(rows)


def write_markdown(
    corr: pd.DataFrame,
    vif: pd.DataFrame,
    budget: int,
    per_run: pd.DataFrame,
    summary: pd.DataFrame,
    tests: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Reviewer Supplementary Experiments")
    lines.append("")
    lines.append("## Scope")
    lines.append(
        "These experiments were run on the current repository and current local dataset "
        "(2,029 observations; 123 firms for model training/calibration; 32 firms for portfolio validation)."
    )
    lines.append("")
    lines.append("## Multicollinearity Check")
    lines.append("Pairwise correlations among the five core indicators:")
    lines.append("")
    lines.append("```text")
    lines.append(corr.to_string())
    lines.append("```")
    lines.append("")
    lines.append("Variance inflation factors:")
    lines.append("")
    lines.append("```text")
    lines.append(vif.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## Optimization Robustness and Benchmarking")
    lines.append(f"Matched evaluation budget for all algorithms: `{budget}` objective evaluations per run.")
    lines.append("")
    lines.append("Summary statistics (mean ± sd across seeds):")
    lines.append("")
    lines.append("```text")
    lines.append(summary.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("Pairwise Mann-Whitney tests comparing SA-NA against each baseline:")
    lines.append("")
    lines.append("```text")
    lines.append(tests.to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("Per-run results:")
    lines.append("")
    lines.append("```text")
    lines.append(per_run.to_string(index=False))
    lines.append("```")
    lines.append("")
    (OUTDIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    cfg, df_full, df_opt, pd_values, ri_scores = load_pipeline()

    corr, vif = multicollinearity_report(df_full, cfg)
    corr.to_csv(OUTDIR / "multicollinearity_correlation.csv")
    vif.to_csv(OUTDIR / "multicollinearity_vif.csv", index=False)

    budget = MATCHED_EVAL_BUDGET

    seeds = list(range(1, 11))
    results = []
    results.extend(run_sana_series("SA-NA", seeds, budget, cfg, df_opt, ri_scores, pd_values, enable_na=True))
    results.extend(run_sana_series("SA-only", seeds, budget, cfg, df_opt, ri_scores, pd_values, enable_na=False))
    for alg in ["NSGA-II", "MOEA/D", "SPEA2"]:
        results.extend(run_pymoo_series(alg, seeds, budget, cfg, df_opt, pd_values))

    assign_front_metrics(results)

    per_run = results_frame(results)
    summary = summary_table(per_run)
    tests = pairwise_tests(per_run, ["SA-only", "NSGA-II", "MOEA/D", "SPEA2"])

    per_run.to_csv(OUTDIR / "per_run_results.csv", index=False)
    summary.to_csv(OUTDIR / "summary_results.csv", index=False)
    tests.to_csv(OUTDIR / "statistical_tests.csv", index=False)

    metadata = {
        "dataset_rows": int(len(df_full)),
        "optimization_firms": int(len(df_opt)),
        "budget_objective_calls": int(budget),
        "algorithms": sorted(per_run["algorithm"].unique().tolist()),
        "seeds": seeds,
    }
    (OUTDIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    write_markdown(corr, vif, budget, per_run, summary, tests)

    print(OUTDIR)
    print(f"Budget: {budget}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
