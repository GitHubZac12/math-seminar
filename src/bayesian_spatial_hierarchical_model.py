from __future__ import annotations

import argparse
import json
from pathlib import Path, PureWindowsPath
from typing import Sequence

import numpy as np
import pandas as pd

from .bayesian_nb_model import _standardize_matrix, _standardize_matrix_using_stats
from .spatial_utils import load_adjacency_matrix


def write_bayesian_spatial_config(
    output_dir: Path,
    geo_id_col: str,
    outcome_col: str,
    population_col: str,
    feature_cols: Sequence[str],
    control_cols: Sequence[str] = (),
    inclusion_col: str = "model_inclusion_flag",
    bayesian_input_csv: str = "bayesian_nb_input.csv",
    adjacency_matrix_csv: str = "spatial_adjacency_matrix.csv",
    adjacency_metadata_json: str = "spatial_adjacency_metadata.json",
) -> Path:
    """Write the config for the Bayesian spatial hierarchical model."""
    output_dir = Path(output_dir)
    config = {
        "geo_id_col": geo_id_col,
        "outcome_col": outcome_col,
        "population_col": population_col,
        "feature_cols": list(feature_cols),
        "control_cols": list(control_cols),
        "inclusion_col": inclusion_col,
        "bayesian_input_csv": bayesian_input_csv,
        "adjacency_matrix_csv": adjacency_matrix_csv,
        "adjacency_metadata_json": adjacency_metadata_json,
        "model": "Bayesian spatial hierarchical negative-binomial disease-mapping model",
        "spatial_prior": "BYM-style area effect: proper CAR structured spatial random effect plus iid unstructured municipal random effect",
        "likelihood": "NegativeBinomial(mu, alpha) with female-population offset",
        "linear_predictor": "log(mu_i) = log(pop_i) + intercept + exposure_beta * X_i + control_gamma * C_i + spatial_structured_i + spatial_unstructured_i",
        "priors": {
            "intercept": "Normal(log(crude rate), 2)",
            "exposure_coefficients": "Normal(0, 0.5), features standardized internally on likelihood rows",
            "control_coefficients": "Normal(0, 1.0), controls standardized internally on likelihood rows",
            "car_structured_raw": "CAR(mu=0, W, alpha=0.95, tau=1), centered to mean zero",
            "structured_spatial_scale": "HalfNormal(0.5)",
            "unstructured_raw": "Normal(0, 1), centered to mean zero",
            "unstructured_spatial_scale": "HalfNormal(0.5)",
            "overdispersion_alpha": "HalfNormal(2)",
        },
        "standardization_scope": "likelihood rows only; the same training statistics are applied to all prediction rows",
        "missing_outcome_policy": "Rows with inclusion_col == 0 are retained for posterior prediction but excluded from the likelihood.",
        "notes": [
            "This is now a true Bayesian spatial hierarchical model because it includes a latent areal spatial process.",
            "The CAR component borrows residual-risk information across neighboring municipalities defined by the adjacency matrix.",
            "The iid component captures municipality-specific residual heterogeneity that is not spatially structured.",
        ],
    }
    path = output_dir / "bayesian_spatial_model_config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def _portable_basename(path: Path | str) -> str:
    text = str(path)
    if "\\" in text:
        return PureWindowsPath(text).name
    return Path(text).name


def validate_spatial_inputs(
    df: pd.DataFrame,
    config: dict,
    adjacency_matrix_path: Path,
) -> dict:
    geo_id_col = config["geo_id_col"]
    required = [
        geo_id_col,
        config["outcome_col"],
        config["population_col"],
        config["inclusion_col"],
        *config.get("feature_cols", []),
        *config.get("control_cols", []),
    ]
    missing = [c for c in required if c not in df.columns]
    geo_ids = df[geo_id_col].astype(str).tolist() if geo_id_col in df.columns else []
    result = {
        "passed": False,
        "n_rows": int(len(df)),
        "n_likelihood_rows": int(pd.to_numeric(df.get(config.get("inclusion_col", ""), pd.Series([])), errors="coerce").fillna(0).sum()) if config.get("inclusion_col") in df.columns else 0,
        "missing_columns": missing,
        "adjacency_matrix_path": _portable_basename(adjacency_matrix_path),
    }
    if missing:
        return result
    try:
        W = load_adjacency_matrix(adjacency_matrix_path, geo_ids, geo_id_col=geo_id_col)
        degrees = W.sum(axis=1)
        result.update(
            {
                "adjacency_shape": list(W.shape),
                "adjacency_symmetric": bool(np.all(W == W.T)),
                "adjacency_diagonal_zero": bool(np.all(np.diag(W) == 0)),
                "min_neighbors": int(degrees.min()),
                "max_neighbors": int(degrees.max()),
                "mean_neighbors": float(degrees.mean()),
                "n_edges": int(W.sum() // 2),
                "passed": bool((degrees > 0).all()),
            }
        )
    except Exception as exc:
        result["adjacency_error"] = str(exc)
    return result


def fit_bayesian_spatial_hierarchical_nb(
    df: pd.DataFrame,
    config: dict,
    adjacency_matrix_path: Path,
    draws: int = 1000,
    tune: int = 1000,
    target_accept: float = 0.92,
    random_seed: int = 42,
    chains: int = 2,
    cores: int = 1,
    top_k: int = 10,
    car_alpha: float = 0.95,
):
    """Fit a BYM-style Bayesian spatial hierarchical negative-binomial model.

    The model uses a proper CAR structured spatial random effect plus an iid
    municipality-level random effect. Rows excluded from the likelihood are still
    predicted using their covariates, population, and spatial position.
    """
    try:
        import arviz as az
        import pymc as pm
        import pytensor.tensor as pt
    except Exception as exc:
        raise ImportError("Install PyMC and ArviZ first: pip install pymc arviz") from exc

    geo_id_col = config["geo_id_col"]
    outcome_col = config["outcome_col"]
    population_col = config["population_col"]
    inclusion_col = config["inclusion_col"]
    feature_cols = list(config.get("feature_cols", []))
    control_cols = list(config.get("control_cols", []))

    work = df.copy()
    work[geo_id_col] = work[geo_id_col].astype(str)
    geo_ids = work[geo_id_col].astype(str).tolist()
    W = load_adjacency_matrix(adjacency_matrix_path, geo_ids, geo_id_col=geo_id_col).astype("int64")
    n_area = len(work)

    include = pd.to_numeric(work[inclusion_col], errors="coerce").fillna(0).astype(bool).to_numpy()
    train_idx = np.where(include)[0].astype("int64")
    train = work.iloc[train_idx].copy()

    X_train, feature_stats = _standardize_matrix(train, feature_cols)
    X_all = _standardize_matrix_using_stats(work, feature_cols, feature_stats)

    C_train = np.empty((len(train), 0))
    C_all = np.empty((len(work), 0))
    control_stats = {}
    if control_cols:
        C_train, control_stats = _standardize_matrix(train, control_cols)
        C_all = _standardize_matrix_using_stats(work, control_cols, control_stats)

    y = pd.to_numeric(train[outcome_col], errors="coerce").to_numpy(dtype="int64")
    pop_train = pd.to_numeric(train[population_col], errors="coerce").to_numpy(dtype="float64")
    pop_all = pd.to_numeric(work[population_col], errors="coerce").to_numpy(dtype="float64")

    crude_rate = max(float(np.nansum(y) / np.nansum(pop_train)), 1e-9)

    coords = {
        "area": geo_ids,
        "feature": feature_cols,
        "control": control_cols,
    }

    with pm.Model(coords=coords) as model:
        intercept = pm.Normal("intercept", mu=np.log(crude_rate), sigma=2.0)
        beta = pm.Normal("beta", mu=0.0, sigma=0.5, shape=len(feature_cols))
        if control_cols:
            gamma = pm.Normal("gamma", mu=0.0, sigma=1.0, shape=len(control_cols))
        else:
            gamma = None

        # Structured spatial effect. A proper CAR prior is faster and more stable
        # than an intrinsic CAR in many small-project environments, while still
        # explicitly borrowing information across adjacent municipalities.
        car_alpha = float(np.clip(car_alpha, 1e-6, 0.999))
        structured_raw = pm.CAR(
            "structured_raw",
            mu=np.zeros(n_area),
            W=W,
            alpha=car_alpha,
            tau=1.0,
            shape=n_area,
        )
        structured_centered = structured_raw - pt.mean(structured_raw)
        structured_scale = pm.HalfNormal("structured_scale", sigma=0.5)
        structured_effect = pm.Deterministic("structured_effect", structured_scale * structured_centered)

        # Unstructured municipal heterogeneity, centered for intercept identifiability.
        unstructured_raw = pm.Normal("unstructured_raw", mu=0.0, sigma=1.0, shape=n_area)
        unstructured_centered = unstructured_raw - pt.mean(unstructured_raw)
        unstructured_scale = pm.HalfNormal("unstructured_scale", sigma=0.5)
        unstructured_effect = pm.Deterministic("unstructured_effect", unstructured_scale * unstructured_centered)

        area_effect = pm.Deterministic("area_effect", structured_effect + unstructured_effect)

        eta_all = intercept + pm.math.dot(X_all, beta) + area_effect + np.log(pop_all)
        if control_cols:
            eta_all = eta_all + pm.math.dot(C_all, gamma)
        eta_train = eta_all[train_idx]

        mu_train = pm.math.exp(eta_train)
        overdispersion_alpha = pm.HalfNormal("overdispersion_alpha", sigma=2.0)
        pm.NegativeBinomial("y_obs", mu=mu_train, alpha=overdispersion_alpha, observed=y)

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

    posterior_rate = trace.posterior["rate_all_per_100k"]
    rate_means = posterior_rate.mean(dim=("chain", "draw")).values
    rate_lows = posterior_rate.quantile(0.025, dim=("chain", "draw")).values
    rate_highs = posterior_rate.quantile(0.975, dim=("chain", "draw")).values

    posterior_mu = trace.posterior["mu_all"]
    count_means = posterior_mu.mean(dim=("chain", "draw")).values

    rate_summary = pd.DataFrame(
        {
            geo_id_col: geo_ids,
            "bayes_spatial_predicted_rate_mean_per_100k": rate_means,
            "bayes_spatial_predicted_rate_ci_lower": rate_lows,
            "bayes_spatial_predicted_rate_ci_upper": rate_highs,
            "bayes_spatial_predicted_count_mean": count_means,
        }
    )

    def summarize_area_var(var_name: str, prefix: str) -> pd.DataFrame:
        arr = trace.posterior[var_name]
        return pd.DataFrame(
            {
                geo_id_col: geo_ids,
                f"{prefix}_mean": arr.mean(dim=("chain", "draw")).values,
                f"{prefix}_ci_lower": arr.quantile(0.025, dim=("chain", "draw")).values,
                f"{prefix}_ci_upper": arr.quantile(0.975, dim=("chain", "draw")).values,
            }
        )

    effect_summary = summarize_area_var("structured_effect", "structured_spatial_effect")
    effect_summary = effect_summary.merge(summarize_area_var("unstructured_effect", "unstructured_spatial_effect"), on=geo_id_col)
    effect_summary = effect_summary.merge(summarize_area_var("area_effect", "combined_area_effect"), on=geo_id_col)

    # Posterior top-K probabilities based on spatial posterior rates.
    # xarray dimension names for unnamed PyMC deterministics can vary by version,
    # so reshape the posterior array directly: (chain, draw, area) -> (sample, area).
    draws_matrix = posterior_rate.values.reshape(-1, n_area)
    k = int(min(max(top_k, 1), n_area))
    top_idx = np.argpartition(-draws_matrix, kth=k - 1, axis=1)[:, :k]
    counts = np.bincount(top_idx.ravel(), minlength=n_area)
    topk_summary = pd.DataFrame(
        {
            geo_id_col: geo_ids,
            "bayes_spatial_top_k_probability": counts / draws_matrix.shape[0],
            "top_k": k,
            "n_posterior_draws": draws_matrix.shape[0],
        }
    ).sort_values("bayes_spatial_top_k_probability", ascending=False)

    stats = {
        "feature_stats": feature_stats,
        "control_stats": control_stats,
        "n_likelihood_rows": int(len(train_idx)),
        "n_prediction_rows": int(len(work)),
        "n_edges": int(W.sum() // 2),
        "min_neighbors": int(W.sum(axis=1).min()),
        "max_neighbors": int(W.sum(axis=1).max()),
        "mean_neighbors": float(W.sum(axis=1).mean()),
        "car_alpha": float(car_alpha),
    }
    return model, trace, rate_summary, effect_summary, topk_summary, stats


def _relative_output_path(path: Path, base: Path) -> str:
    """Return a portable path string for model-status JSON files."""
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except Exception:
        return Path(path).name


def merge_spatial_outputs_with_project(
    project_output_dir: Path,
    posterior_rates: pd.DataFrame,
    topk_summary: pd.DataFrame | None = None,
    geo_id_col: str = "geo_id",
) -> dict:
    """Create map/cube files that include Bayesian spatial posterior summaries."""
    project_output_dir = Path(project_output_dir)
    written: dict = {}
    merged = posterior_rates.copy()
    if topk_summary is not None:
        merged = merged.merge(topk_summary[[geo_id_col, "bayes_spatial_top_k_probability"]], on=geo_id_col, how="left")

    risk_map_path = project_output_dir / "risk_map.geojson"
    if risk_map_path.exists():
        try:
            import geopandas as gpd
            gdf = gpd.read_file(risk_map_path)
            gdf[geo_id_col] = gdf[geo_id_col].astype(str)
            map_out = gdf.merge(merged, on=geo_id_col, how="left")
            out_path = project_output_dir / "risk_map_with_bayesian_spatial.geojson"
            map_out.to_file(out_path, driver="GeoJSON")
            written["risk_map_with_bayesian_spatial"] = _relative_output_path(out_path, project_output_dir)
        except Exception as exc:
            written["risk_map_merge_error"] = str(exc)

    risk_cube_path = project_output_dir / "risk_cube.csv"
    if risk_cube_path.exists():
        try:
            cube = pd.read_csv(risk_cube_path, dtype={geo_id_col: str})
            long_frames = [cube]
            for metric in [
                "bayes_spatial_predicted_rate_mean_per_100k",
                "bayes_spatial_predicted_rate_ci_lower",
                "bayes_spatial_predicted_rate_ci_upper",
                "bayes_spatial_predicted_count_mean",
            ]:
                if metric in posterior_rates.columns:
                    temp = posterior_rates[[geo_id_col, metric]].rename(columns={metric: "value"})
                    temp["metric"] = metric
                    temp["source"] = "bayesian_spatial_hierarchical_nb"
                    long_frames.append(temp)
            if topk_summary is not None and "bayes_spatial_top_k_probability" in topk_summary.columns:
                temp = topk_summary[[geo_id_col, "bayes_spatial_top_k_probability"]].rename(columns={"bayes_spatial_top_k_probability": "value"})
                temp["metric"] = "bayes_spatial_top_k_probability"
                temp["source"] = "bayesian_spatial_hierarchical_nb"
                long_frames.append(temp)
            cube_out = pd.concat(long_frames, ignore_index=True, sort=False)
            out_path = project_output_dir / "risk_cube_with_bayesian_spatial.csv"
            cube_out.to_csv(out_path, index=False)
            written["risk_cube_with_bayesian_spatial"] = _relative_output_path(out_path, project_output_dir)
        except Exception as exc:
            written["risk_cube_merge_error"] = str(exc)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or fit the Bayesian spatial hierarchical negative-binomial model.")
    parser.add_argument("--input-csv", required=True, help="Usually outputs/bayesian_nb_input.csv")
    parser.add_argument("--config-json", required=True, help="Usually outputs/bayesian_spatial_model_config.json")
    parser.add_argument("--output-dir", required=True, help="Directory for Bayesian spatial outputs, e.g. outputs/bayes_spatial")
    parser.add_argument("--fit", action="store_true", help="Actually sample the PyMC model. Without this, only validates inputs.")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--target-accept", type=float, default=0.92)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--car-alpha", type=float, default=0.95, help="Proper CAR autocorrelation parameter. Values close to 1 impose stronger spatial smoothing.")
    parser.add_argument("--merge-project-outputs", action="store_true", help="Write map/cube files with spatial Bayesian summaries into the parent project output directory.")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    config_path = Path(args.config_json)
    config = json.loads(config_path.read_text())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv, dtype={config["geo_id_col"]: str})
    adjacency_path = Path(config.get("adjacency_matrix_csv", "spatial_adjacency_matrix.csv"))
    if not adjacency_path.is_absolute():
        adjacency_path = config_path.parent / adjacency_path

    validation = validate_spatial_inputs(df, config, adjacency_path)
    (out_dir / "spatial_model_input_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if not validation.get("passed"):
        raise ValueError(f"Spatial model input validation failed: {validation}")
    if not args.fit:
        print("Spatial model input validation passed. Re-run with --fit to sample the PyMC model.")
        return

    model, trace, posterior_rates, effect_summary, topk_summary, stats = fit_bayesian_spatial_hierarchical_nb(
        df=df,
        config=config,
        adjacency_matrix_path=adjacency_path,
        draws=args.draws,
        tune=args.tune,
        target_accept=args.target_accept,
        chains=args.chains,
        cores=args.cores,
        top_k=args.top_k,
        car_alpha=args.car_alpha,
    )

    posterior_rates.to_csv(out_dir / "bayesian_spatial_posterior_rates.csv", index=False)
    effect_summary.to_csv(out_dir / "bayesian_spatial_effect_summary.csv", index=False)
    topk_summary.to_csv(out_dir / "bayesian_spatial_top_k_probabilities.csv", index=False)
    (out_dir / "bayesian_spatial_standardization_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    try:
        import arviz as az
        az.to_netcdf(trace, out_dir / "bayesian_spatial_trace.nc")
        az.summary(
            trace,
            var_names=[
                "intercept",
                "beta",
                "gamma",
                "structured_scale",
                "unstructured_scale",
                "overdispersion_alpha",
            ],
            filter_vars="like",
        ).to_csv(out_dir / "bayesian_spatial_trace_summary.csv")
        diagnostics = {
            "fit_completed": True,
            "draws": int(args.draws),
            "tune": int(args.tune),
            "chains": int(args.chains),
            "n_prediction_rows": int(len(df)),
            "n_likelihood_rows": int(stats["n_likelihood_rows"]),
        }
        try:
            sampler = trace.sample_stats
            diagnostics["divergences"] = int(sampler["diverging"].sum().values) if "diverging" in sampler else None
            diagnostics["mean_acceptance_rate"] = float(sampler["acceptance_rate"].mean().values) if "acceptance_rate" in sampler else None
            diagnostics["max_tree_depth"] = int(sampler["tree_depth"].max().values) if "tree_depth" in sampler else None
        except Exception as exc:
            diagnostics["sampler_diagnostics_error"] = str(exc)
        if args.merge_project_outputs:
            project_output_dir = input_csv.parent
            diagnostics["merged_project_outputs"] = merge_spatial_outputs_with_project(
                project_output_dir=project_output_dir,
                posterior_rates=posterior_rates,
                topk_summary=topk_summary,
                geo_id_col=config["geo_id_col"],
            )
        (out_dir / "bayesian_spatial_model_status.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    except Exception as exc:
        (out_dir / "bayesian_spatial_trace_summary_error.txt").write_text(str(exc), encoding="utf-8")

    print(f"Wrote Bayesian spatial outputs to: {out_dir}")


if __name__ == "__main__":
    main()
