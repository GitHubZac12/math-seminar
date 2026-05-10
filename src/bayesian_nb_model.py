
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def write_bayesian_nb_inputs(
    df: pd.DataFrame,
    output_dir: Path,
    geo_id_col: str,
    outcome_col: str,
    population_col: str,
    feature_cols: Sequence[str],
    control_cols: Sequence[str] = (),
    inclusion_col: str = "model_inclusion_flag",
) -> tuple[Path, Path]:
    """Write a clean table/config that can be passed to the PyMC model.

    The main pipeline already standardizes exposure features before fitting the
    frequentist NB model, but this export intentionally leaves raw feature columns
    plus a config list so the Bayesian script can standardize inside the model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = [geo_id_col, outcome_col, population_col, inclusion_col, *feature_cols, *control_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for Bayesian input export: {missing}")

    out = df[required].copy()
    out.to_csv(output_dir / "bayesian_nb_input.csv", index=False)

    config = {
        "geo_id_col": geo_id_col,
        "outcome_col": outcome_col,
        "population_col": population_col,
        "feature_cols": list(feature_cols),
        "control_cols": list(control_cols),
        "inclusion_col": inclusion_col,
        "model": "Bayesian negative binomial with population offset and weakly regularizing priors",
        "priors": {
            "intercept": "Normal(log(crude rate), 2)",
            "exposure_coefficients": "Normal(0, 0.5), features standardized internally",
            "control_coefficients": "Normal(0, 1.0)",
            "overdispersion_alpha": "HalfNormal(2)",
        },
        "standardization_scope": "likelihood rows only; the same training statistics are applied to all prediction rows",
        "note": "Rows with inclusion_col == 0 are retained for prediction but excluded from the likelihood.",
    }
    config_path = output_dir / "bayesian_model_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return output_dir / "bayesian_nb_input.csv", config_path


def _standardize_matrix(df: pd.DataFrame, cols: Sequence[str]):
    """Fit standardization stats on the supplied dataframe and transform it."""
    X = df[list(cols)].copy()
    stats = {}
    for c in cols:
        s = pd.to_numeric(X[c], errors="coerce")
        fill = float(s.median(skipna=True)) if s.notna().any() else 0.0
        filled = s.fillna(fill).astype(float)
        mean = float(filled.mean())
        std = float(filled.std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        X[c] = (filled - mean) / std
        stats[c] = {"fill_value": fill, "mean": mean, "std": std}
    return X.to_numpy(), stats


def _standardize_matrix_using_stats(df: pd.DataFrame, cols: Sequence[str], stats: dict):
    """Transform a dataframe using already-fitted standardization stats.

    This is important for rows excluded from the likelihood but retained for
    prediction: their features must be scaled using training-row statistics, not
    statistics recomputed using all rows.
    """
    X = df[list(cols)].copy()
    for c in cols:
        if c not in stats:
            raise KeyError(f"Missing standardization stats for column {c}")
        s = pd.to_numeric(X[c], errors="coerce")
        fill = stats[c].get("fill_value", 0.0)
        mean = stats[c].get("mean", 0.0)
        std = stats[c].get("std", 1.0) or 1.0
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        X[c] = (s.fillna(fill).astype(float) - mean) / std
    return X.to_numpy()


def fit_bayesian_negative_binomial(
    df: pd.DataFrame,
    geo_id_col: str,
    outcome_col: str,
    population_col: str,
    feature_cols: Sequence[str],
    control_cols: Sequence[str] = (),
    inclusion_col: str = "model_inclusion_flag",
    draws: int = 1000,
    tune: int = 1000,
    target_accept: float = 0.9,
    random_seed: int = 42,
    chains: int = 2,
    cores: int = 1,
):
    """Fit the Bayesian NB model with PyMC.

    This function requires:
        pip install pymc arviz

    It returns (model, trace, posterior_prediction_summary).
    """
    try:
        import pymc as pm
        import arviz as az
    except Exception as exc:
        raise ImportError("Install PyMC and ArviZ first: pip install pymc arviz") from exc

    work = df.copy()
    include = pd.to_numeric(work[inclusion_col], errors="coerce").fillna(0).astype(bool)
    train = work.loc[include].copy()

    X_train, feature_stats = _standardize_matrix(train, feature_cols)
    X_all = _standardize_matrix_using_stats(work, feature_cols, feature_stats)

    C_train = np.empty((len(train), 0))
    C_all = np.empty((len(work), 0))
    control_stats = {}
    if control_cols:
        C_train, control_stats = _standardize_matrix(train, control_cols)
        C_all = _standardize_matrix_using_stats(work, control_cols, control_stats)

    y = pd.to_numeric(train[outcome_col], errors="coerce").to_numpy()
    pop_train = pd.to_numeric(train[population_col], errors="coerce").to_numpy()
    pop_all = pd.to_numeric(work[population_col], errors="coerce").to_numpy()

    crude_rate = max(float(np.nansum(y) / np.nansum(pop_train)), 1e-9)

    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=np.log(crude_rate), sigma=2.0)
        beta = pm.Normal("beta", mu=0.0, sigma=0.5, shape=X_train.shape[1])
        if C_train.shape[1] > 0:
            gamma = pm.Normal("gamma", mu=0.0, sigma=1.0, shape=C_train.shape[1])
            eta = intercept + pm.math.dot(X_train, beta) + pm.math.dot(C_train, gamma) + np.log(pop_train)
            eta_all = intercept + pm.math.dot(X_all, beta) + pm.math.dot(C_all, gamma) + np.log(pop_all)
        else:
            eta = intercept + pm.math.dot(X_train, beta) + np.log(pop_train)
            eta_all = intercept + pm.math.dot(X_all, beta) + np.log(pop_all)

        mu = pm.math.exp(eta)
        alpha = pm.HalfNormal("overdispersion_alpha", sigma=2.0)
        pm.NegativeBinomial("y_obs", mu=mu, alpha=alpha, observed=y)

        mu_all = pm.Deterministic("mu_all", pm.math.exp(eta_all))
        rate_all_per_100k = pm.Deterministic("rate_all_per_100k", mu_all / pop_all * 100000.0)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            random_seed=random_seed,
            chains=chains,
            cores=cores,
        )

    posterior = trace.posterior["rate_all_per_100k"]
    means = posterior.mean(dim=("chain", "draw")).values
    lows = posterior.quantile(0.025, dim=("chain", "draw")).values
    highs = posterior.quantile(0.975, dim=("chain", "draw")).values
    summary = pd.DataFrame(
        {
            geo_id_col: work[geo_id_col].astype(str).values,
            "bayes_predicted_rate_mean_per_100k": means,
            "bayes_predicted_rate_ci_lower": lows,
            "bayes_predicted_rate_ci_upper": highs,
        }
    )
    return model, trace, summary, {"feature_stats": feature_stats, "control_stats": control_stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or fit a Bayesian negative-binomial model.")
    parser.add_argument("--input-csv", required=True, help="Path to bayesian_nb_input.csv")
    parser.add_argument("--config-json", required=True, help="Path to bayesian_model_config.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit", action="store_true", help="Actually fit the PyMC model. Without this, only validates input/config.")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--target-accept", type=float, default=0.9)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    config = json.loads(Path(args.config_json).read_text())
    df = pd.read_csv(input_csv, dtype={config["geo_id_col"]: str})
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    required = [config["geo_id_col"], config["outcome_col"], config["population_col"], config["inclusion_col"], *config["feature_cols"], *config.get("control_cols", [])]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Bayesian input is missing columns: {missing}")

    if not args.fit:
        (out_dir / "bayesian_input_validation.json").write_text(
            json.dumps({"passed": True, "n_rows": len(df), "n_likelihood_rows": int(pd.to_numeric(df[config["inclusion_col"]]).sum())}, indent=2),
            encoding="utf-8",
        )
        print("Input validation passed. Re-run with --fit to sample the PyMC model.")
        return

    model, trace, summary, stats = fit_bayesian_negative_binomial(
        df=df,
        geo_id_col=config["geo_id_col"],
        outcome_col=config["outcome_col"],
        population_col=config["population_col"],
        feature_cols=config["feature_cols"],
        control_cols=config.get("control_cols", []),
        inclusion_col=config["inclusion_col"],
        draws=args.draws,
        tune=args.tune,
        target_accept=args.target_accept,
        chains=args.chains,
        cores=args.cores,
    )
    summary.to_csv(out_dir / "bayesian_nb_posterior_rates.csv", index=False)
    try:
        import arviz as az
        az.to_netcdf(trace, out_dir / "bayesian_nb_trace.nc")
        az.summary(trace, var_names=["intercept", "beta", "gamma", "overdispersion_alpha"], filter_vars="like").to_csv(
            out_dir / "bayesian_trace_summary.csv"
        )
    except Exception as exc:
        (out_dir / "bayesian_trace_summary_error.txt").write_text(str(exc), encoding="utf-8")
    (out_dir / "bayesian_standardization_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    diagnostics = {
        "fit_completed": True,
        "draws": int(args.draws),
        "tune": int(args.tune),
        "chains": int(args.chains),
        "cores": int(args.cores),
        "target_accept": float(args.target_accept),
        "standardization_scope": "likelihood-row stats applied to all prediction rows",
        "n_prediction_rows": int(len(summary)),
        "n_likelihood_rows": int(pd.to_numeric(df[config["inclusion_col"]], errors="coerce").fillna(0).sum()),
    }
    try:
        sampler = trace.sample_stats
        diagnostics["divergences"] = int(sampler["diverging"].sum().values) if "diverging" in sampler else None
        diagnostics["mean_acceptance_rate"] = float(sampler["acceptance_rate"].mean().values) if "acceptance_rate" in sampler else None
        diagnostics["max_tree_depth"] = int(sampler["tree_depth"].max().values) if "tree_depth" in sampler else None
    except Exception as exc:
        diagnostics["sampler_diagnostics_error"] = str(exc)
    (out_dir / "bayesian_output_status.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"Wrote Bayesian outputs to {out_dir}")


if __name__ == "__main__":
    main()
