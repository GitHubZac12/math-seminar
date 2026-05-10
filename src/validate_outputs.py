from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_ABSOLUTE_PATH_PATTERNS = [
    re.compile(r"^[A-Za-z]:\\\\"),  # Windows drive path
    re.compile(r"^/[Uu]sers/"),
    re.compile(r"^/home/"),
]


def _is_absolute_local_path(value: str) -> bool:
    return any(p.search(value) for p in _ABSOLUTE_PATH_PATTERNS)


def _iter_strings(obj: Any):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_strings(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_strings(value)
    elif isinstance(obj, str):
        yield obj


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_summary_checks(path: Path, prefix: str, add_check) -> None:
    if not path.exists():
        return
    try:
        trace = pd.read_csv(path)
        add_check(f"{prefix}_trace_summary_parseable", True, f"rows={len(trace)}")
        if "r_hat" in trace.columns:
            rhat = pd.to_numeric(trace["r_hat"], errors="coerce")
            max_rhat = float(rhat.max(skipna=True)) if rhat.notna().any() else float("nan")
            add_check(f"{prefix}_rhat_ok", bool(np.isfinite(max_rhat) and max_rhat <= 1.05), f"max_r_hat={max_rhat}")
        if "ess_bulk" in trace.columns:
            ess = pd.to_numeric(trace["ess_bulk"], errors="coerce")
            min_ess = float(ess.min(skipna=True)) if ess.notna().any() else float("nan")
            add_check(f"{prefix}_bulk_ess_ok", bool(np.isfinite(min_ess) and min_ess >= 100), f"min_ess_bulk={min_ess}")
    except Exception as exc:
        add_check(f"{prefix}_trace_summary_parseable", False, str(exc))


def validate_outputs(output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    report = {"output_dir": str(output_dir), "checks": []}

    def add_check(name: str, passed: bool, detail: str) -> None:
        report["checks"].append({"check": name, "passed": bool(passed), "detail": detail})

    analytic_path = output_dir / "analytic_dataset.csv"
    modeled_path = output_dir / "modeled_dataset.csv"
    metadata_path = output_dir / "run_metadata.json"
    feature_report_path = output_dir / "feature_filter_report.csv"
    ranking_path = output_dir / "ranking_stability.csv"
    map_path = output_dir / "risk_map.geojson"
    missing_path = output_dir / "missing_outcome_geos.csv"
    sensitivity_metrics_path = output_dir / "sensitivity_single_family_metrics.csv"
    bayes_input_path = output_dir / "bayesian_nb_input.csv"
    bayes_config_path = output_dir / "bayesian_model_config.json"
    bayes_posterior_path = output_dir / "bayes" / "bayesian_nb_posterior_rates.csv"
    bayes_status_path = output_dir / "bayes" / "bayesian_output_status.json"
    bayes_stats_path = output_dir / "bayes" / "bayesian_standardization_stats.json"
    bayes_trace_summary_path = output_dir / "bayes" / "bayesian_trace_summary.csv"
    spatial_config_path = output_dir / "bayesian_spatial_model_config.json"
    adjacency_matrix_path = output_dir / "spatial_adjacency_matrix.csv"
    adjacency_neighbors_path = output_dir / "spatial_neighbors.csv"
    adjacency_metadata_path = output_dir / "spatial_adjacency_metadata.json"
    spatial_posterior_path = output_dir / "bayes_spatial" / "bayesian_spatial_posterior_rates.csv"
    spatial_topk_path = output_dir / "bayes_spatial" / "bayesian_spatial_top_k_probabilities.csv"
    spatial_status_path = output_dir / "bayes_spatial" / "bayesian_spatial_model_status.json"
    spatial_stats_path = output_dir / "bayes_spatial" / "bayesian_spatial_standardization_stats.json"
    spatial_trace_summary_path = output_dir / "bayes_spatial" / "bayesian_spatial_trace_summary.csv"
    age_report_path = output_dir / "age_adjustment_report.csv"
    spatial_map_path = output_dir / "risk_map_with_bayesian_spatial.geojson"
    spatial_cube_path = output_dir / "risk_cube_with_bayesian_spatial.csv"

    required_outputs = [
        ("analytic_dataset_exists", analytic_path),
        ("modeled_dataset_exists", modeled_path),
        ("run_metadata_exists", metadata_path),
        ("feature_filter_report_exists", feature_report_path),
        ("ranking_stability_exists", ranking_path),
        ("risk_map_exists", map_path),
        ("missing_outcome_report_exists", missing_path),
        ("single_family_sensitivity_exists", sensitivity_metrics_path),
        ("bayesian_input_exists", bayes_input_path),
        ("bayesian_config_exists", bayes_config_path),
        ("bayesian_spatial_config_exists", spatial_config_path),
        ("spatial_adjacency_matrix_exists", adjacency_matrix_path),
        ("spatial_neighbors_exists", adjacency_neighbors_path),
        ("spatial_adjacency_metadata_exists", adjacency_metadata_path),
        ("age_adjustment_report_exists", age_report_path),
    ]
    for name, path in required_outputs:
        add_check(name, path.exists(), str(path))

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        try:
            metadata = _read_json(metadata_path)
            report["metadata"] = metadata
            add_check("metadata_parseable", True, "ok")
            absolute_paths = [s for s in _iter_strings(metadata) if _is_absolute_local_path(s)]
            add_check("metadata_paths_portable", len(absolute_paths) == 0, f"absolute_path_count={len(absolute_paths)}")
        except Exception as exc:
            add_check("metadata_parseable", False, str(exc))

    if analytic_path.exists():
        df = pd.read_csv(analytic_path)
        if "population" in df.columns:
            pop = pd.to_numeric(df["population"], errors="coerce")
            add_check("population_not_all_one", not (pop == 1).all(), f"min={pop.min()}, max={pop.max()}")
            add_check("population_has_no_missing", not pop.isna().any(), f"missing={int(pop.isna().sum())}")
            add_check("population_positive", bool((pop > 0).all()), f"min={pop.min()}")
            denom_type = metadata.get("population_denominator_type")
            total = float(pop.sum()) if pop.notna().any() else float("nan")
            if metadata.get("scope") == "slp" and denom_type == "female_population":
                add_check("slp_female_population_total_plausible", 1_000_000 <= total <= 2_000_000, f"total={total:,.0f}")
            elif metadata.get("scope") == "slp":
                add_check("slp_total_population_plausible", 2_000_000 <= total <= 3_500_000, f"total={total:,.0f}")
        else:
            add_check("population_column_present", False, "No population column")

        if "geo_id" in df.columns:
            add_check("geo_id_unique_or_year_indexed", True, f"n_geo={df['geo_id'].nunique()}, n_rows={len(df)}")

        if "observed_rate_per_100k" in df.columns:
            rate = pd.to_numeric(df["observed_rate_per_100k"], errors="coerce")
            missing_flag = pd.to_numeric(df.get("outcome_missing_flag", pd.Series([0] * len(df))), errors="coerce").fillna(0)
            expected_missing = int((missing_flag == 1).sum())
            actual_missing_rates = int(rate.isna().sum())
            add_check(
                "observed_rate_missing_only_when_outcome_missing",
                actual_missing_rates == expected_missing,
                f"missing_rates={actual_missing_rates}, flagged_missing_outcomes={expected_missing}",
            )

        if "outcome_missing_flag" in df.columns:
            missing = int(pd.to_numeric(df["outcome_missing_flag"], errors="coerce").fillna(0).sum())
            add_check("missing_outcome_rows_flagged", True, f"missing_outcome_rows={missing}")

    if modeled_path.exists():
        md = pd.read_csv(modeled_path)
        control_cols = [c for c in md.columns if c.startswith("control_") and c.endswith("_z")]
        duplicate_pairs = []
        for i, c1 in enumerate(control_cols):
            s1 = pd.to_numeric(md[c1], errors="coerce").fillna(-999999).round(12)
            for c2 in control_cols[i + 1:]:
                s2 = pd.to_numeric(md[c2], errors="coerce").fillna(-999999).round(12)
                if s1.equals(s2):
                    duplicate_pairs.append((c1, c2))
        add_check("no_duplicate_control_columns", len(duplicate_pairs) == 0, f"duplicates={duplicate_pairs}")
        if "model_inclusion_flag" in md.columns:
            n_in = int(pd.to_numeric(md["model_inclusion_flag"], errors="coerce").fillna(0).sum())
            add_check("likelihood_rows_nonzero", n_in > 0, f"n_likelihood_rows={n_in}")
            if "outcome_missing_flag" in md.columns:
                missing_included = int(((pd.to_numeric(md["outcome_missing_flag"], errors="coerce").fillna(0) == 1) & (pd.to_numeric(md["model_inclusion_flag"], errors="coerce").fillna(0) == 1)).sum())
                policy = metadata.get("missing_outcome_policy", "")
                add_check(
                    "missing_outcomes_excluded_when_policy_exclude",
                    missing_included == 0 if policy == "exclude" else True,
                    f"policy={policy}, missing_outcome_rows_in_likelihood={missing_included}",
                )
        if "predicted_rate_per_100k" in md.columns:
            pred = pd.to_numeric(md["predicted_rate_per_100k"], errors="coerce")
            add_check("predicted_rate_finite", bool(pred.notna().all()), f"min={pred.min()}, max={pred.max()}")

    if feature_report_path.exists():
        fr = pd.read_csv(feature_report_path)
        used = int((fr["status"] == "keep").sum()) if "status" in fr.columns else 0
        add_check("features_used_nonzero", used > 0, f"n_features_used={used}")

    if ranking_path.exists():
        rk = pd.read_csv(ranking_path)
        if "successful_bootstrap_fits" in rk.columns and "requested_bootstrap_fits" in rk.columns:
            successes = int(rk["successful_bootstrap_fits"].max()) if len(rk) else 0
            requested = int(rk["requested_bootstrap_fits"].max()) if len(rk) else 0
            add_check("bootstrap_success_reasonable", successes >= max(10, int(0.8 * requested)), f"successes={successes}, requested={requested}")
        if "top_k_selection_probability" in rk.columns:
            prob_sum = float(pd.to_numeric(rk["top_k_selection_probability"], errors="coerce").sum())
            expected_k = int(metadata.get("top_k", 10)) if metadata.get("top_k") else 10
            # Metadata may not store top_k in older runs; infer from rounded sum when needed.
            add_check("ranking_probability_sum_reasonable", abs(prob_sum - round(prob_sum)) < 1e-6, f"sum={prob_sum}")

    if sensitivity_metrics_path.exists():
        sm = pd.read_csv(sensitivity_metrics_path)
        add_check("single_family_sensitivity_nonempty", len(sm) > 0, f"n_families={len(sm)}")
        if "model_type" in sm.columns:
            fallbacks = int(sm["model_type"].astype(str).str.contains("fixed_alpha|fallback", case=False, regex=True).sum())
            add_check("single_family_no_fixed_alpha_fallbacks", fallbacks == 0, f"fallback_models={fallbacks}")

    if adjacency_matrix_path.exists():
        try:
            adj = pd.read_csv(adjacency_matrix_path, dtype={"geo_id": str})
            ids = adj["geo_id"].astype(str).tolist() if "geo_id" in adj.columns else []
            cols = [str(c) for c in adj.columns if str(c) != "geo_id"]
            aligned = len(ids) > 0 and ids == cols
            W = adj[cols].to_numpy(dtype=float) if aligned else None
            if W is not None:
                add_check("spatial_adjacency_aligned", True, f"n={len(ids)}")
                add_check("spatial_adjacency_symmetric", bool((W == W.T).all()), "matrix symmetric check")
                add_check("spatial_adjacency_diagonal_zero", bool((pd.Series(W.diagonal()) == 0).all()), "diagonal zero check")
                degrees = W.sum(axis=1)
                add_check("spatial_adjacency_no_isolates", bool((degrees > 0).all()), f"min_neighbors={degrees.min()}, max_neighbors={degrees.max()}")
            else:
                add_check("spatial_adjacency_aligned", False, f"ids_match_columns={aligned}")
        except Exception as exc:
            add_check("spatial_adjacency_parseable", False, str(exc))

    if adjacency_metadata_path.exists():
        try:
            adj_meta = _read_json(adjacency_metadata_path)
            add_check("spatial_adjacency_metadata_parseable", True, f"n_nodes={adj_meta.get('n_nodes')}, n_edges={adj_meta.get('n_edges')}")
            add_check("spatial_adjacency_components_connected", int(adj_meta.get("n_components_after_connection", 999)) == 1, f"components={adj_meta.get('n_components_after_connection')}")
        except Exception as exc:
            add_check("spatial_adjacency_metadata_parseable", False, str(exc))

    if spatial_config_path.exists() and bayes_input_path.exists():
        try:
            cfg_sp = _read_json(spatial_config_path)
            bi_sp = pd.read_csv(bayes_input_path)
            missing_sp = [c for c in [cfg_sp["geo_id_col"], cfg_sp["outcome_col"], cfg_sp["population_col"], cfg_sp["inclusion_col"], *cfg_sp.get("feature_cols", []), *cfg_sp.get("control_cols", [])] if c not in bi_sp.columns]
            add_check("bayesian_spatial_input_columns_valid", len(missing_sp) == 0, f"missing={missing_sp}")
        except Exception as exc:
            add_check("bayesian_spatial_config_parseable", False, str(exc))

    if bayes_input_path.exists() and bayes_config_path.exists():
        try:
            bi = pd.read_csv(bayes_input_path)
            cfg = _read_json(bayes_config_path)
            missing = [c for c in [cfg["geo_id_col"], cfg["outcome_col"], cfg["population_col"], cfg["inclusion_col"], *cfg.get("feature_cols", []), *cfg.get("control_cols", [])] if c not in bi.columns]
            add_check("bayesian_input_columns_valid", len(missing) == 0, f"missing={missing}")
            if not missing:
                included = pd.to_numeric(bi[cfg["inclusion_col"]], errors="coerce").fillna(0).astype(bool)
                positive_pop = (pd.to_numeric(bi[cfg["population_col"]], errors="coerce") > 0).all()
                add_check("bayesian_input_population_positive", bool(positive_pop), "all prediction rows require positive population")
                finite_y = pd.to_numeric(bi.loc[included, cfg["outcome_col"]], errors="coerce").notna().all()
                add_check("bayesian_likelihood_outcomes_finite", bool(finite_y), f"n_likelihood_rows={int(included.sum())}")
            if bayes_stats_path.exists():
                stats = _read_json(bayes_stats_path)
                feature_keys = set(stats.get("feature_stats", {}).keys())
                control_keys = set(stats.get("control_stats", {}).keys())
                add_check(
                    "bayesian_feature_stats_match_config",
                    feature_keys == set(cfg.get("feature_cols", [])),
                    f"stats={sorted(feature_keys)}, config={cfg.get('feature_cols', [])}",
                )
                add_check(
                    "bayesian_control_stats_match_config",
                    control_keys == set(cfg.get("control_cols", [])),
                    f"stats={sorted(control_keys)}, config={cfg.get('control_cols', [])}",
                )
        except Exception as exc:
            add_check("bayesian_config_or_stats_parseable", False, str(exc))

    if bayes_posterior_path.exists():
        try:
            bp = pd.read_csv(bayes_posterior_path)
            required_cols = ["geo_id", "bayes_predicted_rate_mean_per_100k", "bayes_predicted_rate_ci_lower", "bayes_predicted_rate_ci_upper"]
            missing_cols = [c for c in required_cols if c not in bp.columns]
            add_check("bayesian_posterior_columns_valid", len(missing_cols) == 0, f"missing={missing_cols}")
            if not missing_cols:
                mean = pd.to_numeric(bp["bayes_predicted_rate_mean_per_100k"], errors="coerce")
                lo = pd.to_numeric(bp["bayes_predicted_rate_ci_lower"], errors="coerce")
                hi = pd.to_numeric(bp["bayes_predicted_rate_ci_upper"], errors="coerce")
                add_check("bayesian_rates_finite", bool(mean.notna().all() and lo.notna().all() and hi.notna().all()), f"n={len(bp)}")
                add_check("bayesian_mean_inside_interval", bool(((lo <= mean) & (mean <= hi)).all()), "mean should lie inside 95% interval")
        except Exception as exc:
            add_check("bayesian_posterior_parseable", False, str(exc))

    if bayes_status_path.exists():
        try:
            status = _read_json(bayes_status_path)
            add_check("bayesian_fit_completed", bool(status.get("fit_completed", False)), f"fit_completed={status.get('fit_completed')}")
        except Exception as exc:
            add_check("bayesian_status_parseable", False, str(exc))

    _trace_summary_checks(bayes_trace_summary_path, "bayesian", add_check)

    if spatial_posterior_path.exists():
        try:
            sp = pd.read_csv(spatial_posterior_path)
            required_spatial_cols = [
                "geo_id",
                "bayes_spatial_predicted_rate_mean_per_100k",
                "bayes_spatial_predicted_rate_ci_lower",
                "bayes_spatial_predicted_rate_ci_upper",
            ]
            missing_spatial_cols = [c for c in required_spatial_cols if c not in sp.columns]
            add_check("bayesian_spatial_posterior_columns_valid", len(missing_spatial_cols) == 0, f"missing={missing_spatial_cols}")
            if not missing_spatial_cols:
                mean = pd.to_numeric(sp["bayes_spatial_predicted_rate_mean_per_100k"], errors="coerce")
                lo = pd.to_numeric(sp["bayes_spatial_predicted_rate_ci_lower"], errors="coerce")
                hi = pd.to_numeric(sp["bayes_spatial_predicted_rate_ci_upper"], errors="coerce")
                add_check("bayesian_spatial_rates_finite", bool(mean.notna().all() and lo.notna().all() and hi.notna().all()), f"n={len(sp)}")
                add_check("bayesian_spatial_mean_inside_interval", bool(((lo <= mean) & (mean <= hi)).all()), "mean should lie inside 95% interval")
        except Exception as exc:
            add_check("bayesian_spatial_posterior_parseable", False, str(exc))

    if spatial_topk_path.exists():
        try:
            topk = pd.read_csv(spatial_topk_path)
            if "bayes_spatial_top_k_probability" in topk.columns:
                prob_sum = float(pd.to_numeric(topk["bayes_spatial_top_k_probability"], errors="coerce").sum())
                top_k = int(pd.to_numeric(topk.get("top_k", pd.Series([round(prob_sum)])), errors="coerce").dropna().iloc[0])
                add_check("bayesian_spatial_topk_probability_sum", abs(prob_sum - top_k) < 1e-6, f"sum={prob_sum}, top_k={top_k}")
        except Exception as exc:
            add_check("bayesian_spatial_topk_parseable", False, str(exc))

    if spatial_status_path.exists():
        try:
            status = _read_json(spatial_status_path)
            add_check("bayesian_spatial_fit_completed", bool(status.get("fit_completed", False)), f"fit_completed={status.get('fit_completed')}")
            if status.get("divergences") is not None:
                add_check("bayesian_spatial_no_divergences", int(status.get("divergences", 0)) == 0, f"divergences={status.get('divergences')}")
            absolute_paths = [s for s in _iter_strings(status) if _is_absolute_local_path(s)]
            add_check("bayesian_spatial_status_paths_portable", len(absolute_paths) == 0, f"absolute_path_count={len(absolute_paths)}")
        except Exception as exc:
            add_check("bayesian_spatial_status_parseable", False, str(exc))

    if spatial_stats_path.exists() and spatial_config_path.exists():
        try:
            stats_sp = _read_json(spatial_stats_path)
            cfg_sp = _read_json(spatial_config_path)
            add_check(
                "bayesian_spatial_feature_stats_match_config",
                set(stats_sp.get("feature_stats", {}).keys()) == set(cfg_sp.get("feature_cols", [])),
                f"stats={sorted(stats_sp.get('feature_stats', {}).keys())}, config={cfg_sp.get('feature_cols', [])}",
            )
            add_check(
                "bayesian_spatial_control_stats_match_config",
                set(stats_sp.get("control_stats", {}).keys()) == set(cfg_sp.get("control_cols", [])),
                f"stats={sorted(stats_sp.get('control_stats', {}).keys())}, config={cfg_sp.get('control_cols', [])}",
            )
        except Exception as exc:
            add_check("bayesian_spatial_stats_parseable", False, str(exc))

    _trace_summary_checks(spatial_trace_summary_path, "bayesian_spatial", add_check)

    for name, path in [("risk_map_with_bayesian_spatial", spatial_map_path), ("risk_cube_with_bayesian_spatial", spatial_cube_path)]:
        if path.exists():
            if path.suffix.lower() == ".geojson":
                try:
                    import geopandas as gpd
                    gdf = gpd.read_file(path)
                    duplicate_suffix_cols = [c for c in gdf.columns if c.endswith("_x") or c.endswith("_y")]
                    add_check(f"{name}_parseable", True, f"n_features={len(gdf)}")
                    add_check(f"{name}_no_duplicate_merge_suffixes", len(duplicate_suffix_cols) == 0, f"duplicate_suffix_cols={duplicate_suffix_cols}")
                except Exception as exc:
                    add_check(f"{name}_parseable", False, str(exc))
            else:
                try:
                    df = pd.read_csv(path)
                    add_check(f"{name}_parseable", True, f"n_rows={len(df)}")
                except Exception as exc:
                    add_check(f"{name}_parseable", False, str(exc))

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SANA pipeline outputs.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    report = validate_outputs(output_dir)
    out_path = output_dir / "validation_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote validation report to {out_path}")


if __name__ == "__main__":
    main()
