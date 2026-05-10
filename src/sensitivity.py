
from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from .modeling import fit_negative_binomial, predict_counts


def infer_exposure_family(feature_name: str) -> str:
    """Infer source family from engineered feature names.

    Examples:
    - ambientalagua_kernel_intensity -> ambientalagua
    - ambientalretc_nearest_source_m -> ambientalretc
    - ambientalretc_count_within_5000m -> ambientalretc
    """
    for marker in ["_kernel_intensity", "_nearest_source_m", "_count_within_"]:
        if marker in feature_name:
            return feature_name.split(marker)[0]
    return re.split(r"__", feature_name)[0]


def _safe_model_metric(result, name: str):
    value = getattr(result, name, np.nan)
    try:
        return float(value)
    except Exception:
        return np.nan


def run_single_family_sensitivity(
    training_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    geo_id_col: str,
    outcome_col: str,
    population_col: str,
    feature_cols: Sequence[str],
    control_cols: Sequence[str] = (),
    top_k: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit one model per exposure family.

    This helps diagnose whether multi-pollutant conclusions are driven by correlated
    exposure families. The outputs are intentionally simple and dashboard/report-ready.
    """
    families = {}
    for col in feature_cols:
        families.setdefault(infer_exposure_family(col), []).append(col)

    summary_frames: List[pd.DataFrame] = []
    metric_rows = []
    ranking_frames: List[pd.DataFrame] = []

    for family, cols in sorted(families.items()):
        try:
            fit = fit_negative_binomial(
                training_df,
                outcome_col=outcome_col,
                population_col=population_col,
                feature_cols=cols,
                control_cols=control_cols,
            )
            summary = fit.summary_table.copy()
            summary.insert(0, "exposure_family", family)
            summary.insert(1, "family_features", ", ".join(cols))
            summary_frames.append(summary)

            preds = predict_counts(
                fit,
                prediction_df,
                outcome_col=outcome_col,
                population_col=population_col,
                control_cols=control_cols,
            )
            pred_rank = preds.copy()
            pred_rank[geo_id_col] = prediction_df.loc[pred_rank["row_index"], geo_id_col].astype(str).values
            pred_rank = pred_rank.sort_values("predicted_rate_per_100k", ascending=False).head(top_k)
            pred_rank.insert(0, "exposure_family", family)
            ranking_frames.append(pred_rank[[ "exposure_family", geo_id_col, "predicted_count", "predicted_rate_per_100k"]])

            metric_rows.append(
                {
                    "exposure_family": family,
                    "n_features": len(cols),
                    "features": ", ".join(cols),
                    "model_type": fit.model_type,
                    "aic": _safe_model_metric(fit.model, "aic"),
                    "bic": _safe_model_metric(fit.model, "bic"),
                    "llf": _safe_model_metric(fit.model, "llf"),
                    "fit_warnings": " | ".join(fit.fit_warnings),
                    "top_k_geo_ids": ", ".join(pred_rank[geo_id_col].astype(str).tolist()),
                }
            )
        except Exception as exc:
            metric_rows.append(
                {
                    "exposure_family": family,
                    "n_features": len(cols),
                    "features": ", ".join(cols),
                    "model_type": "failed",
                    "aic": np.nan,
                    "bic": np.nan,
                    "llf": np.nan,
                    "fit_warnings": str(exc),
                    "top_k_geo_ids": "",
                }
            )

    summary_out = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    metrics_out = pd.DataFrame(metric_rows).sort_values(["model_type", "aic"], na_position="last")
    rankings_out = pd.concat(ranking_frames, ignore_index=True) if ranking_frames else pd.DataFrame()
    return summary_out, metrics_out, rankings_out
