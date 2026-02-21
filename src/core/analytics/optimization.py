"""Parameter optimization using Optuna (Bayesian search by default)."""
import math
from typing import Any, Callable


class ParameterOptimizer:
    """
    Runs parameter optimization via Optuna.
    Each trial runs a full backtest and evaluates the objective metric.
    """
    def __init__(self, method: str = "bayesian", objective: str = "sharpe_ratio", n_trials: int = 100):
        self.method = method
        self.objective = objective
        self.n_trials = n_trials

    def optimize(
        self,
        run_backtest_fn: Callable[[dict], dict],  # params → metrics dict
        param_space: dict,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> dict:
        """
        Run optimization. Returns best params, best value, and all trial results.
        param_space format: {"fast_period": {"type": "int", "low": 5, "high": 50}, ...}
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise RuntimeError("optuna not installed. Run: pip install optuna")

        sampler = self._make_sampler()
        study = optuna.create_study(direction="maximize", sampler=sampler)
        trial_results = []

        def objective_fn(trial):
            params = self._suggest_params(trial, param_space)
            try:
                metrics = run_backtest_fn(params)
                value = metrics.get(self.objective, 0.0) or 0.0
                trial_results.append({"params": params, "value": value, "metrics": metrics})
                if progress_callback:
                    progress_callback(trial.number + 1, self.n_trials, value)
                return value
            except Exception:
                return -999.0

        study.optimize(objective_fn, n_trials=self.n_trials)

        # Overfitting score: compare top performers to overall distribution
        overfitting_score = self._compute_overfitting_score(study, trial_results)

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "objective": self.objective,
            "n_trials": self.n_trials,
            "overfitting_score": overfitting_score,
            "trials": trial_results,
        }

    def _make_sampler(self):
        import optuna
        if self.method == "bayesian":
            return optuna.samplers.TPESampler()
        elif self.method == "grid":
            return optuna.samplers.GridSampler()
        else:
            return optuna.samplers.RandomSampler()

    def _suggest_params(self, trial, param_space: dict) -> dict:
        params = {}
        for name, spec in param_space.items():
            ptype = spec.get("type", "float")
            if ptype == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"])
            elif ptype == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
        return params

    def _compute_overfitting_score(self, study, trial_results: list) -> float:
        """
        Estimate overfitting by comparing top 10% trial performance to bottom 90%.
        High variance in results → higher overfitting risk.
        """
        if len(trial_results) < 10:
            return 0.0
        values = sorted([t["value"] for t in trial_results], reverse=True)
        top_mean = sum(values[:max(1, len(values)//10)]) / max(1, len(values)//10)
        all_mean = sum(values) / len(values)
        if all_mean <= 0:
            return 1.0
        return min(1.0, max(0.0, (top_mean - all_mean) / abs(top_mean) if top_mean != 0 else 0))
