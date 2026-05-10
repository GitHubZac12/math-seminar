from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import warnings as pywarnings

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class ModelResults:
    model: object
    summary_table: pd.DataFrame
    design_columns: List[str]
    feature_columns: List[str]
    feature_stats: Dict[str, Dict[str, float]]
    fitted: pd.Series
    ranking_table: pd.DataFrame
    model_type: str
    fit_warnings: List[str] = field(default_factory=list)


def diagnose_feature_columns(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    min_nonzero: int = 5,
    corr_threshold: float = 0.98,
) -> pd.DataFrame:
    """Return a table explaining which exposure features are kept or dropped.

    Rules:
    1. drop missing/non-numeric/constant features;
    2. drop features with fewer than `min_nonzero` non-zero municipalities;
    3. drop exact duplicate feature vectors;
    4. greedily drop highly correlated features.
    """
    rows = []
    numeric = pd.DataFrame(index=df.index)

    for col in feature_cols:
        if col not in df.columns:
            rows.append({"feature": col, "status": "drop", "reason": "missing_column", "nonzero_count": 0, "nunique": 0})
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        numeric[col] = s
        nonzero_count = int(((s.fillna(0) != 0)).sum())
        nunique = int(s.nunique(dropna=True))
        if s.notna().sum() == 0:
            rows.append({"feature": col, "status": "drop", "reason": "all_missing", "nonzero_count": nonzero_count, "nunique": nunique})
        elif nunique <= 1:
            rows.append({"feature": col, "status": "drop", "reason": "constant_or_all_zero", "nonzero_count": nonzero_count, "nunique": nunique})
        elif nonzero_count < min_nonzero:
            rows.append({"feature": col, "status": "drop", "reason": f"too_sparse_nonzero_lt_{min_nonzero}", "nonzero_count": nonzero_count, "nunique": nunique})
        else:
            rows.append({"feature": col, "status": "candidate", "reason": "candidate", "nonzero_count": nonzero_count, "nunique": nunique})

    report = pd.DataFrame(rows)
    if report.empty:
        return report

    candidates = report.loc[report["status"] == "candidate", "feature"].tolist()
    kept: List[str] = []
    seen_signatures = {}

    for col in candidates:
        s = numeric[col]
        fill_value = float(s.median(skipna=True)) if s.notna().any() else 0.0
        signature = tuple(np.round(s.fillna(fill_value).astype(float).to_numpy(), 12))
        if signature in seen_signatures:
            report.loc[report["feature"] == col, ["status", "reason"]] = ["drop", f"exact_duplicate_of_{seen_signatures[signature]}"]
            continue
        seen_signatures[signature] = col
        kept.append(col)

    # Greedy high-correlation filtering. Keep the feature that appeared earlier in the list.
    final_kept: List[str] = []
    for col in kept:
        s = numeric[col].astype(float)
        fill_value = float(s.median(skipna=True)) if s.notna().any() else 0.0
        s = s.fillna(fill_value)
        drop_reason = None
        for prev in final_kept:
            prev_s = numeric[prev].astype(float)
            prev_fill = float(prev_s.median(skipna=True)) if prev_s.notna().any() else 0.0
            prev_s = prev_s.fillna(prev_fill)
            if s.std(ddof=0) == 0 or prev_s.std(ddof=0) == 0:
                continue
            corr = float(np.corrcoef(s, prev_s)[0, 1])
            if np.isfinite(corr) and abs(corr) >= corr_threshold:
                drop_reason = f"high_corr_{corr:.3f}_with_{prev}"
                break
        if drop_reason:
            report.loc[report["feature"] == col, ["status", "reason"]] = ["drop", drop_reason]
        else:
            final_kept.append(col)
            report.loc[report["feature"] == col, ["status", "reason"]] = ["keep", "passes_filters"]

    return report


def filter_feature_columns(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    min_nonzero: int = 5,
    corr_threshold: float = 0.98,
) -> List[str]:
    report = diagnose_feature_columns(df, feature_cols, min_nonzero=min_nonzero, corr_threshold=corr_threshold)
    if report.empty:
        return []
    return report.loc[report["status"] == "keep", "feature"].tolist()


def _build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    X = df[list(feature_cols)].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    stats: Dict[str, Dict[str, float]] = {}
    for c in X.columns:
        s = X[c]
        if feature_stats is None:
            fill_value = float(s.median(skipna=True)) if s.notna().any() else 0.0
            filled = s.fillna(fill_value).astype(float)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            if not np.isfinite(std) or std <= 0:
                std = 1.0
            stats[c] = {"fill_value": fill_value, "mean": mean, "std": std}
        else:
            if c not in feature_stats:
                raise KeyError(f"Missing standardization stats for feature {c}")
            stats[c] = feature_stats[c]
            fill_value = stats[c]["fill_value"]
            mean = stats[c]["mean"]
            std = stats[c]["std"] if stats[c]["std"] > 0 else 1.0
            filled = s.fillna(fill_value).astype(float)
        X[c] = (filled - stats[c]["mean"]) / stats[c]["std"]

    return X, stats


def prepare_model_matrix(
    df: pd.DataFrame,
    outcome_col: str,
    population_col: str,
    feature_cols: Sequence[str],
    control_cols: Sequence[str] = (),
    feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Dict[str, Dict[str, float]]]:
    if not feature_cols:
        raise ValueError("No feature columns were provided for modeling.")

    cols = [outcome_col, population_col, *feature_cols, *control_cols]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for modeling: {missing}")

    work = df[cols].copy()
    work[outcome_col] = pd.to_numeric(work[outcome_col], errors="coerce")
    work[population_col] = pd.to_numeric(work[population_col], errors="coerce")
    work = work.dropna(subset=[outcome_col, population_col]).copy()
    work = work[work[population_col] > 0].copy()

    y = work[outcome_col]
    offset = np.log(work[population_col])

    X_features, stats = _build_feature_matrix(work, feature_cols, feature_stats=feature_stats)

    X_controls = pd.DataFrame(index=work.index)
    for c in control_cols:
        X_controls[c] = pd.to_numeric(work[c], errors="coerce")
        fill = float(X_controls[c].median(skipna=True)) if X_controls[c].notna().any() else 0.0
        X_controls[c] = X_controls[c].fillna(fill)

    X = pd.concat([X_features, X_controls], axis=1)
    X = sm.add_constant(X, has_constant="add")

    mask = y.notna() & np.isfinite(offset) & np.isfinite(X).all(axis=1)
    return X.loc[mask], y.loc[mask], offset.loc[mask], stats


def _fit_discrete_negative_binomial(y: pd.Series, X: pd.DataFrame, offset: pd.Series):
    model = sm.NegativeBinomial(y, X, offset=offset)
    return model.fit(disp=False, maxiter=1000)


def _fit_glm_negative_binomial(y: pd.Series, X: pd.DataFrame, offset: pd.Series):
    model = sm.GLM(y, X, family=sm.families.NegativeBinomial(), offset=offset)
    return model.fit(maxiter=500)


def _predict_result(result, X: pd.DataFrame, offset: pd.Series) -> pd.Series:
    try:
        pred = result.predict(X, offset=offset)
    except TypeError:
        pred = result.predict(X)
    return pd.Series(np.asarray(pred), index=X.index, name="predicted_count")


def _summarize_result(result, model_type: str) -> pd.DataFrame:
    params = pd.Series(result.params)
    bse = pd.Series(result.bse).reindex(params.index)
    tvals = pd.Series(getattr(result, "tvalues", np.nan), index=params.index) if not isinstance(getattr(result, "tvalues", np.nan), float) else pd.Series(np.nan, index=params.index)
    pvals = pd.Series(result.pvalues).reindex(params.index)
    try:
        conf = result.conf_int()
        if isinstance(conf, pd.DataFrame):
            conf = conf.copy()
            conf.columns = ["ci_lower", "ci_upper"]
        else:
            conf = pd.DataFrame(conf, index=params.index, columns=["ci_lower", "ci_upper"])
    except Exception:
        conf = pd.DataFrame({"ci_lower": np.nan, "ci_upper": np.nan}, index=params.index)

    out = pd.DataFrame(
        {
            "term": params.index,
            "coef": params.values,
            "std_err": bse.values,
            "z": tvals.values,
            "p_value": pvals.values,
            "model_type": model_type,
        }
    ).merge(conf.reset_index().rename(columns={"index": "term"}), on="term", how="left")
    out["exp_coef"] = np.exp(out["coef"].where(out["term"] != "alpha"))
    out["note"] = np.where(out["term"].isin(["const", "alpha"]), "", "coef is per 1 SD increase in standardized feature")
    return out


def fit_negative_binomial(
    df: pd.DataFrame,
    outcome_col: str,
    population_col: str,
    feature_cols: Sequence[str],
    control_cols: Sequence[str] = (),
    feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> ModelResults:
    X, y, offset, stats = prepare_model_matrix(
        df,
        outcome_col=outcome_col,
        population_col=population_col,
        feature_cols=feature_cols,
        control_cols=control_cols,
        feature_stats=feature_stats,
    )
    warnings: List[str] = []
    try:
        with pywarnings.catch_warnings(record=True) as caught:
            pywarnings.simplefilter("always")
            result = _fit_discrete_negative_binomial(y, X, offset)
        caught_messages = [str(w.message) for w in caught]
        converged = bool(getattr(result, "mle_retvals", {}).get("converged", True))
        bse = pd.Series(getattr(result, "bse", []))
        if not converged:
            raise RuntimeError("Discrete NB optimizer did not converge")
        if len(bse) and bse.isna().any():
            raise RuntimeError("Discrete NB Hessian/bse contains NaN")
        warnings.extend([f"Discrete NB warning: {m}" for m in caught_messages])
        model_type = "discrete_negative_binomial_estimated_dispersion"
    except Exception as exc:
        warnings.append(f"Discrete NB failed; fell back to GLM NB with fixed alpha=1. Reason: {exc}")
        with pywarnings.catch_warnings(record=True) as caught:
            pywarnings.simplefilter("always")
            result = _fit_glm_negative_binomial(y, X, offset)
        warnings.extend([f"GLM NB warning: {str(w.message)}" for w in caught])
        model_type = "glm_negative_binomial_fixed_alpha"

    summary_table = _summarize_result(result, model_type=model_type)
    fitted = _predict_result(result, X, offset)
    ranking = pd.DataFrame(
        {
            "row_index": X.index,
            "predicted_count": fitted,
            "predicted_rate_per_100k": fitted / np.exp(offset) * 100000.0,
        }
    ).sort_values("predicted_rate_per_100k", ascending=False)

    return ModelResults(
        model=result,
        summary_table=summary_table,
        design_columns=list(X.columns),
        feature_columns=list(feature_cols),
        feature_stats=stats,
        fitted=fitted,
        ranking_table=ranking,
        model_type=model_type,
        fit_warnings=warnings,
    )


def predict_counts(
    fit: ModelResults,
    df: pd.DataFrame,
    outcome_col: str,
    population_col: str,
    control_cols: Sequence[str] = (),
) -> pd.DataFrame:
    X, _, offset, _ = prepare_model_matrix(
        df,
        outcome_col=outcome_col,
        population_col=population_col,
        feature_cols=fit.feature_columns,
        control_cols=control_cols,
        feature_stats=fit.feature_stats,
    )
    preds = _predict_result(fit.model, X, offset)
    return pd.DataFrame(
        {
            "row_index": X.index,
            "predicted_count": preds,
            "predicted_rate_per_100k": preds / np.exp(offset) * 100000.0,
        }
    )


def bootstrap_ranking_stability(
    df: pd.DataFrame,
    geo_id_col: str,
    outcome_col: str,
    population_col: str,
    feature_cols: Sequence[str],
    control_cols: Sequence[str] = (),
    top_k: int = 10,
    n_boot: int = 200,
    random_state: int = 42,
    prediction_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Estimate top-K ranking stability using coefficient draws.

    Earlier versions repeatedly refit a discrete negative-binomial model on
    bootstrap resamples. That is slow and can be numerically fragile on small
    municipal datasets. This implementation fits the model once, draws
    coefficients from the asymptotic covariance matrix, and recomputes the
    top-K list on each draw. It preserves the same output fields while making
    the diagnostic much more stable and fast.
    """
    rng = np.random.default_rng(random_state)
    if geo_id_col not in df.columns:
        raise KeyError(f"{geo_id_col} not in dataframe")
    if not feature_cols:
        raise ValueError("No feature columns available for ranking stability.")

    train = df.reset_index(drop=True).copy()
    predict_base = prediction_df.reset_index(drop=True).copy() if prediction_df is not None else train.copy()
    if geo_id_col not in predict_base.columns:
        raise KeyError(f"{geo_id_col} not in prediction dataframe")

    fit = fit_negative_binomial(
        train,
        outcome_col=outcome_col,
        population_col=population_col,
        feature_cols=feature_cols,
        control_cols=control_cols,
    )

    X_all, _, offset_all, _ = prepare_model_matrix(
        predict_base,
        outcome_col=outcome_col,
        population_col=population_col,
        feature_cols=fit.feature_columns,
        control_cols=control_cols,
        feature_stats=fit.feature_stats,
    )
    design_cols = list(X_all.columns)

    params = pd.Series(fit.model.params)
    beta_hat = params.reindex(design_cols).astype(float)
    if beta_hat.isna().any():
        missing = beta_hat[beta_hat.isna()].index.tolist()
        raise ValueError(f"Model parameters missing design columns: {missing}")

    try:
        cov = fit.model.cov_params()
        if not isinstance(cov, pd.DataFrame):
            cov = pd.DataFrame(cov, index=params.index, columns=params.index)
        cov = cov.reindex(index=design_cols, columns=design_cols).astype(float)
    except Exception:
        cov = pd.DataFrame(np.diag(np.square(np.repeat(0.05, len(design_cols)))), index=design_cols, columns=design_cols)

    cov = cov.replace([np.inf, -np.inf], np.nan)
    if cov.isna().any().any():
        diag = np.square(pd.Series(getattr(fit.model, "bse", 0.05), index=params.index).reindex(design_cols).fillna(0.05).astype(float))
        cov = pd.DataFrame(np.diag(np.maximum(diag.to_numpy(), 1e-8)), index=design_cols, columns=design_cols)

    cov_np = (cov.to_numpy() + cov.to_numpy().T) / 2.0
    # Ensure positive semidefiniteness for sampling.
    eigvals = np.linalg.eigvalsh(cov_np)
    min_eig = float(np.min(eigvals)) if len(eigvals) else 0.0
    if min_eig < 1e-10:
        cov_np = cov_np + np.eye(cov_np.shape[0]) * (1e-10 - min_eig)

    counts = {str(g): 0 for g in predict_base.loc[X_all.index, geo_id_col].astype(str)}
    successes = 0
    X_np = X_all.to_numpy(dtype=float)
    offset_np = offset_all.to_numpy(dtype=float)
    geo_ids = predict_base.loc[X_all.index, geo_id_col].astype(str).to_numpy()

    for _ in range(n_boot):
        try:
            beta_draw = rng.multivariate_normal(beta_hat.to_numpy(), cov_np, method="svd")
            eta = X_np @ beta_draw + offset_np
            rate = np.exp(np.clip(eta - offset_np, -30, 30)) * 100000.0
            top_idx = np.argsort(rate)[::-1][:top_k]
        except Exception:
            continue
        for g in geo_ids[top_idx]:
            counts[g] = counts.get(str(g), 0) + 1
        successes += 1

    denom = max(successes, 1)
    out = pd.DataFrame(
        {
            geo_id_col: list(counts.keys()),
            "top_k_selection_probability": [v / denom for v in counts.values()],
            "successful_bootstrap_fits": successes,
            "requested_bootstrap_fits": n_boot,
            "ranking_stability_method": "asymptotic_coefficient_draws",
        }
    ).sort_values("top_k_selection_probability", ascending=False)
    return out

