
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from .exposure_features import ExposureConfig, compute_point_source_exposure, points_from_tabular
from .io_utils import (
    build_geo_key,
    choose_population_column,
    discover_files,
    extract_digits,
    infer_geo_columns,
    load_spatial,
    load_tabular,
    normalize_name,
    safe_numeric,
)
from .modeling import (
    bootstrap_ranking_stability,
    diagnose_feature_columns,
    fit_negative_binomial,
    predict_counts,
)
from .risk_cube import build_risk_cube, export_geo_outputs
from .sensitivity import run_single_family_sensitivity
from .bayesian_nb_model import write_bayesian_nb_inputs
from .spatial_utils import write_adjacency_outputs
from .bayesian_spatial_hierarchical_model import write_bayesian_spatial_config


SCOPE_STATE_CODES = {"slp": "24", "merida": "31"}


def _metadata_path(path: Path | str, base: Path | None = None) -> str:
    """Return a portable path string for metadata and reports.

    Absolute local paths are useful while running, but they make packaged outputs
    noisy and non-portable. When a base directory is available, store paths
    relative to it; otherwise fall back to the file name.
    """
    path = Path(path)
    if base is not None:
        try:
            return str(path.resolve().relative_to(Path(base).resolve()))
        except Exception:
            pass
    return path.name


def _metadata_file_table(files: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    """Return a copy of the file inventory with portable paths for output."""
    out = files.copy()
    if "path" in out.columns:
        out["path"] = out["path"].map(lambda x: _metadata_path(x, repo_root))
    if "parent" in out.columns:
        out["parent"] = out["parent"].map(lambda x: _metadata_path(x, repo_root))
    return out


def infer_scope_from_path(path: Path) -> str:
    p = str(path).lower()
    if "datos slp" in p or "san luis" in p or "24-slp" in p or "24-san luis" in p:
        return "slp"
    if "merida" in p or "yuc" in p or "31-yuc" in p:
        return "merida"
    return "national"


def filter_files_for_scope(files: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "national":
        return files.copy()
    scope_patterns = {
        "slp": r"datos slp|san luis|24-slp|24-san luis|iter[_-]?24",
        "merida": r"merida|yuc|31-yuc|iter[_-]?31",
    }
    pat = scope_patterns.get(scope)
    if not pat:
        return files.copy()
    mask = files["path"].str.contains(pat, case=False, na=False)
    scoped = files[mask].copy()
    return scoped if not scoped.empty else files.copy()


def _file_has_expected_state(path: Path, expected_state_code: Optional[str], spatial: bool = False) -> bool:
    if expected_state_code is None:
        return True
    try:
        df = load_spatial(path) if spatial else load_tabular(path)
    except Exception:
        return False
    cols = infer_geo_columns(df)
    if cols.get("cve_ent") and cols["cve_ent"] in df.columns:
        state_codes = extract_digits(df[cols["cve_ent"]]).str.zfill(2).dropna().unique()
        return expected_state_code in set(state_codes)
    if "cvegeo" in df.columns:
        state_codes = extract_digits(df["cvegeo"]).str[:2].dropna().unique()
        return expected_state_code in set(state_codes)
    return False


def choose_polygon_file(files: pd.DataFrame, scope: Optional[str] = None) -> Optional[Path]:
    spatial = files[files["suffix"].isin([".shp", ".geojson", ".gpkg", ".json"])].copy()
    if spatial.empty:
        return None
    if scope:
        spatial = filter_files_for_scope(spatial, scope)
    spatial = spatial[~spatial["name"].str.contains("settings|mexico_by_estado|estado.json", case=False, na=False)]
    expected_state_code = SCOPE_STATE_CODES.get(scope or "")

    scored = []
    for path_str in spatial["path"].values:
        path = Path(path_str)
        score = 0
        lower_path = str(path).lower()
        if scope == "slp" and ("datos slp" in lower_path or "san luis" in lower_path or "24-slp" in lower_path):
            score += 100
        if scope == "merida" and ("merida" in lower_path or "yuc" in lower_path):
            score += 100
        if any(k in lower_path for k in ["municip", "poligono", "geo", "boundary", "shp"]):
            score += 30
        try:
            gdf = load_spatial(path)
            cols = infer_geo_columns(gdf)
            if cols.get("cve_ent") and cols.get("cve_mun"):
                score += 100
            if "cvegeo" in gdf.columns:
                score += 100
            if expected_state_code and _file_has_expected_state(path, expected_state_code, spatial=True):
                score += 150
            tmp = gdf.copy()
            tmp["geo_id"] = build_geo_key(tmp, cols.get("state"), cols.get("municipio"), cols.get("cve_ent"), cols.get("cve_mun"))
            n_geo = tmp["geo_id"].nunique(dropna=True)
            if scope == "slp" and 50 <= n_geo <= 70:
                score += 100
            elif scope == "merida" and 50 <= n_geo <= 120:
                score += 60
            elif n_geo > 10:
                score += 20
        except Exception:
            pass
        scored.append((score, path))

    if not scored:
        return None
    return sorted(scored, reverse=True)[0][1]


def choose_outcome_file(files: pd.DataFrame, scope: Optional[str] = None) -> Optional[Path]:
    candidates = files[files["name"].str.contains("mortal|mort|casos|incid|cama|lla", case=False, na=False)].copy()
    if candidates.empty:
        return None
    candidates = candidates[~candidates["name"].str.contains("ambiental|agua|retc|mina|metal|ladrill", case=False, na=False)]
    if candidates.empty:
        return None
    if scope:
        candidates = filter_files_for_scope(candidates, scope)
    expected_state_code = SCOPE_STATE_CODES.get(scope or "")

    def score_path(path: str) -> int:
        score = 0
        lower_path = path.lower()
        lower_name = Path(path).name.lower()
        if scope == "slp" and ("datos slp" in lower_path or "san luis" in lower_path):
            score += 80
        if scope == "merida" and "merida" in lower_path:
            score += 80
        if any(k in lower_name for k in ["mortal", "mort", "cama", "lla"]):
            score += 50
        if any(k in lower_name for k in ["incid", "casos"]):
            score += 25
        try:
            df = load_tabular(Path(path))
            cols = infer_geo_columns(df)
            if cols["cve_ent"] and cols["cve_mun"]:
                score += 100
            if cols["state"] and cols["municipio"]:
                score += 30
            if cols["mortality"] or cols["incidence"]:
                score += 50
            if expected_state_code and _file_has_expected_state(Path(path), expected_state_code):
                score += 150
        except Exception:
            return score
        return score

    candidates["score"] = candidates["path"].apply(score_path)
    return Path(candidates.sort_values("score", ascending=False).iloc[0]["path"])


def choose_population_file(files: pd.DataFrame, scope: str) -> Optional[Path]:
    candidates = files[files["name"].str.contains("iter|pobl|pob", case=False, na=False)].copy()
    if candidates.empty:
        return None
    state_code = SCOPE_STATE_CODES.get(scope)
    scored = []
    for path_str in candidates["path"].tolist():
        path = Path(path_str)
        score = 0
        lower = str(path).lower()
        name = path.name.lower()
        if "iter" in name:
            score += 100
        if state_code and re.search(fr"iter[_-]?{state_code}\b|{state_code}[_-]", name):
            score += 250
        if scope == "slp" and ("datos nacionales" in lower or "poblacion" in lower):
            score += 40
        if scope == "merida" and ("merida" in lower or "yuc" in lower or "datos nacionales" in lower):
            score += 40
        if state_code and _file_has_expected_state(path, state_code):
            score += 150
        scored.append((score, path))
    return sorted(scored, reverse=True)[0][1]


def choose_exposure_files(files: pd.DataFrame, scope: Optional[str] = None) -> List[Path]:
    candidates = files[files["name"].str.contains("agua|retc|mina|metal|ladr|contam", case=False, na=False)].copy()
    if scope:
        scoped = filter_files_for_scope(candidates, scope)
        if not scoped.empty:
            candidates = scoped
    candidates = candidates[~candidates["name"].str.contains("mortal|mort|casos|incid|analytic|risk|summary|inventory", case=False, na=False)]
    return [Path(p) for p in candidates["path"].tolist()]


def _drop_state_or_country_totals(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> pd.DataFrame:
    out = df.copy()
    if cols.get("cve_mun") and cols["cve_mun"] in out.columns:
        mun_digits = extract_digits(out[cols["cve_mun"]]).str.zfill(3)
        out = out[~mun_digits.isin(["000", "nan"])]
    return out


def _municipality_total_mask(df: pd.DataFrame, cols: Dict[str, Optional[str]]) -> Optional[pd.Series]:
    masks = []
    possible_loc_cols = [cols.get("cve_loc"), "cve_loc", "loc", "clave_localidad", "localidad_code"]
    for col in [c for c in possible_loc_cols if c and c in df.columns]:
        loc = extract_digits(df[col])
        loc4 = loc.str.zfill(4)
        mask = loc.notna() & loc4.eq("0000")
        if 0 < int(mask.sum()) < len(df):
            masks.append((f"{col}_equals_0000", mask))

    possible_name_cols = [cols.get("locality_name"), "nom_loc", "nombre_localidad", "localidad"]
    for col in [c for c in possible_name_cols if c and c in df.columns]:
        text = df[col].astype(str).map(normalize_name)
        mask = text.str.contains(r"total_del_municipio|total_municipal|municipio_total", regex=True, na=False)
        if 0 < int(mask.sum()) < len(df):
            masks.append((f"{col}_municipality_total_name", mask))

    if not masks:
        return None
    return sorted(masks, key=lambda x: int(x[1].sum()), reverse=True)[0][1]


def _population_denominator_type(pop_col: str) -> str:
    n = normalize_name(pop_col)
    if any(k in n for k in ["fem", "mujer", "mujeres"]):
        return "female_population"
    if any(k in n for k in ["masc", "hombre", "hombres"]):
        return "male_population"
    return "total_population"


def _find_first_age_col(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        regex = re.compile(pat)
        for col in df.columns:
            if regex.search(normalize_name(col)):
                return col
    return None


def _add_age_adjustment_features(pop: pd.DataFrame, denominator_col: str) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Add age-structure adjustment covariates when recognizable columns exist.

    This is not a true age-standardized rate unless the outcome is age-specific. It is
    an age-structure-adjusted baseline: the model can control for older female population
    shares when the population workbook has those columns.
    """
    out = pop.copy()
    denom = safe_numeric(out[denominator_col]).replace(0, np.nan)

    candidate_patterns = {
        "age_share_female_50plus": [
            r"(p|pob).*50.*(ymas|y_mas|mas).*(f|fem|muj)",
            r"(f|fem|muj).*(50).*(ymas|y_mas|mas)",
        ],
        "age_share_female_60plus": [
            r"(p|pob).*60.*(ymas|y_mas|mas).*(f|fem|muj)",
            r"(f|fem|muj).*(60).*(ymas|y_mas|mas)",
        ],
        "age_share_female_65plus": [
            r"(p|pob).*65.*(ymas|y_mas|mas).*(f|fem|muj)",
            r"(f|fem|muj).*(65).*(ymas|y_mas|mas)",
        ],
        "age_share_female_15_49": [
            r"(p|pob).*15.*49.*(f|fem|muj)",
            r"(f|fem|muj).*15.*49",
        ],
    }

    added = []
    source_cols = {}
    for feature_name, patterns in candidate_patterns.items():
        col = _find_first_age_col(out, patterns)
        if not col:
            continue
        vals = safe_numeric(out[col]) / denom
        vals = vals.where((vals >= 0) & (vals <= 1.5))
        if vals.notna().sum() == 0 or vals.nunique(dropna=True) <= 1:
            continue
        out[feature_name] = vals
        added.append(feature_name)
        source_cols[feature_name] = col

    # Record a preferred older-age share for interpretation, but do not create
    # a duplicate covariate. Earlier versions added
    # age_share_female_older_priority as a copy of age_share_female_60plus,
    # which made the design matrix rank-deficient.
    older_priority = None
    for preferred in ["age_share_female_65plus", "age_share_female_60plus", "age_share_female_50plus"]:
        if preferred in added:
            older_priority = preferred
            break

    meta = {
        "age_adjustment_strategy": "age_structure_covariates_added" if added else "no_recognized_age_structure_columns",
        "age_adjustment_columns": added,
        "age_adjustment_source_columns": source_cols,
        "age_adjustment_older_priority_column": older_priority,
        "age_adjustment_note": (
            "These are age-structure controls, not direct age-standardized mortality rates. "
            "Direct age standardization requires age-specific outcome counts. "
            "Duplicate age controls are intentionally not added."
        ),
    }
    return out, meta


def build_population_table(pop_path: Path, scope: str, age_adjustment: str = "auto") -> Tuple[pd.DataFrame, Dict[str, object]]:
    pop_raw = load_tabular(pop_path)
    cols = infer_geo_columns(pop_raw)
    pop = pop_raw.copy()
    expected_state_code = SCOPE_STATE_CODES.get(scope)

    if expected_state_code and cols.get("cve_ent") and cols["cve_ent"] in pop.columns:
        state = extract_digits(pop[cols["cve_ent"]]).str.zfill(2)
        pop = pop[state == expected_state_code].copy()

    pop = _drop_state_or_country_totals(pop, cols)
    total_mask = _municipality_total_mask(pop, cols)
    aggregation_strategy = "sum_localities_after_excluding_state_totals"
    if total_mask is not None:
        pop = pop[total_mask].copy()
        aggregation_strategy = "municipality_total_rows_only"

    pop["geo_id"] = build_geo_key(
        pop,
        state_col=cols.get("state"),
        municipio_col=cols.get("municipio"),
        cve_ent_col=cols.get("cve_ent"),
        cve_mun_col=cols.get("cve_mun"),
    )
    pop_col = choose_population_column(pop)
    if pop_col is None:
        raise ValueError(f"Could not infer a population column from {pop_path.name}")

    pop["population"] = safe_numeric(pop[pop_col])
    age_meta: Dict[str, object] = {
        "age_adjustment_strategy": "off",
        "age_adjustment_columns": [],
        "age_adjustment_source_columns": {},
        "age_adjustment_note": "Age adjustment disabled.",
    }
    if age_adjustment == "auto":
        pop, age_meta = _add_age_adjustment_features(pop, pop_col)

    keep_cols = ["geo_id", "population"] + list(age_meta.get("age_adjustment_columns", []))
    out = pop[keep_cols].copy()
    out = out.dropna(subset=["geo_id", "population"])
    out = out[out["population"] > 0]

    agg_spec = {"population": "max" if aggregation_strategy == "municipality_total_rows_only" else "sum"}
    for c in age_meta.get("age_adjustment_columns", []):
        agg_spec[c] = "mean"
    out = out.groupby("geo_id", as_index=False).agg(agg_spec)

    denominator_type = _population_denominator_type(pop_col)
    meta = {
        "population_file": str(pop_path),
        "population_column": pop_col,
        "population_denominator_type": denominator_type,
        "population_rows_raw": int(len(pop_raw)),
        "population_rows_after_filtering": int(len(pop)),
        "population_municipalities": int(out["geo_id"].nunique()),
        "population_aggregation_strategy": aggregation_strategy,
        "population_total_after_filtering": float(out["population"].sum()) if not out.empty else None,
        **age_meta,
    }
    return out, meta


def validate_population(analytic: pd.DataFrame, scope: Optional[str] = None, denominator_type: str = "total_population") -> List[str]:
    warnings: List[str] = []
    if "population" not in analytic.columns:
        raise ValueError("Population column is missing after join.")
    pop = pd.to_numeric(analytic["population"], errors="coerce")
    if pop.isna().all():
        raise ValueError("Population join failed: all population values are missing.")
    missing = int(pop.isna().sum())
    if missing:
        raise ValueError(f"Population join failed for {missing} areas. Check population join keys.")
    if (pop <= 0).any():
        raise ValueError("Population join produced non-positive values for at least one area.")
    if pop.nunique(dropna=True) <= 1:
        raise ValueError("Population join appears invalid: population has no variation.")
    if (pop == 1).mean() > 0.8:
        raise ValueError("Population join appears invalid: most population values equal 1.")

    total = float(pop.sum())
    if scope == "slp":
        if denominator_type == "female_population":
            if not (1_000_000 <= total <= 2_000_000):
                warnings.append(
                    f"SLP female population denominator totals {total:,.0f}; expected roughly 1.0-2.0 million. Verify denominator column and total-row filtering."
                )
        elif not (2_000_000 <= total <= 3_500_000):
            warnings.append(
                f"SLP total population denominator totals {total:,.0f}; expected roughly 2.0-3.5 million. Verify denominator column and total-row filtering."
            )
    if scope == "merida" and total <= 0:
        warnings.append("Mérida/Yucatán population total is non-positive; verify denominator.")
    return warnings


def _build_outcome_table(outcome_path: Path) -> Tuple[pd.DataFrame, Dict[str, object], pd.DataFrame]:
    outcome = load_tabular(outcome_path)
    out_cols = infer_geo_columns(outcome)
    outcome["geo_id"] = build_geo_key(
        outcome,
        state_col=out_cols.get("state"),
        municipio_col=out_cols.get("municipio"),
        cve_ent_col=out_cols.get("cve_ent"),
        cve_mun_col=out_cols.get("cve_mun"),
    )

    outcome_source = None
    if out_cols["mortality"] and out_cols["mortality"] in outcome.columns:
        outcome["outcome_count"] = safe_numeric(outcome[out_cols["mortality"]])
        outcome_source = out_cols["mortality"]
    elif out_cols["incidence"] and out_cols["incidence"] in outcome.columns:
        outcome["outcome_count"] = safe_numeric(outcome[out_cols["incidence"]])
        outcome_source = out_cols["incidence"]
    else:
        numeric_cols = [c for c in outcome.columns if pd.api.types.is_numeric_dtype(outcome[c])]
        numeric_cols = [c for c in numeric_cols if c not in {out_cols.get("cve_ent"), out_cols.get("cve_mun"), out_cols.get("cve_loc")}]
        if not numeric_cols:
            raise ValueError("Could not infer an outcome count column.")
        outcome["outcome_count"] = safe_numeric(outcome[numeric_cols[0]])
        outcome_source = numeric_cols[0]

    audit_cols = [c for c in [out_cols.get("cve_ent"), out_cols.get("cve_mun"), out_cols.get("state"), out_cols.get("municipio")] if c and c in outcome.columns]
    audit = outcome[audit_cols + ["geo_id", "outcome_count"]].copy()
    audit.insert(0, "raw_row_index", outcome.index)
    audit["outcome_key_valid"] = audit["geo_id"].astype(str).str.len().ge(5) & audit["geo_id"].notna()
    audit["outcome_count_valid"] = audit["outcome_count"].notna()

    keep_cols = ["geo_id", "outcome_count"]
    if out_cols["year"] and out_cols["year"] in outcome.columns:
        outcome["year"] = outcome[out_cols["year"]]
        keep_cols.append("year")

    outcome_small = outcome[keep_cols].copy()
    outcome_small = outcome_small.dropna(subset=["geo_id", "outcome_count"])
    outcome_small = outcome_small[outcome_small["geo_id"].astype(str).str.len().ge(5)]
    group_cols = ["geo_id"] + (["year"] if "year" in outcome_small.columns else [])
    outcome_agg = outcome_small.groupby(group_cols, as_index=False).agg({"outcome_count": "sum"})
    meta = {
        "outcome_file": str(outcome_path),
        "outcome_column": outcome_source,
        "outcome_rows_raw": int(len(outcome)),
        "outcome_rows_aggregated": int(len(outcome_agg)),
        "outcome_invalid_key_rows": int((~audit["outcome_key_valid"]).sum()),
        "outcome_missing_count_rows": int((~audit["outcome_count_valid"]).sum()),
        "has_year_dimension": bool("year" in outcome_agg.columns),
    }
    return outcome_agg, meta, audit


def _base_geography_frame(polygons: gpd.GeoDataFrame, pop: pd.DataFrame, years: Optional[List[object]]) -> gpd.GeoDataFrame:
    base = polygons[["geo_id", "geometry"]].drop_duplicates(subset=["geo_id"]).merge(pop, on="geo_id", how="left")
    if years:
        year_df = pd.DataFrame({"year": years})
        base = base.merge(year_df, how="cross")
    return gpd.GeoDataFrame(base, geometry="geometry", crs=polygons.crs)


def build_analytic_dataset(repo_root: Path, output_dir: Path, scope: Optional[str] = None, age_adjustment: str = "auto") -> Tuple[gpd.GeoDataFrame, Dict[str, object]]:
    files = discover_files(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    _metadata_file_table(files, repo_root).to_csv(output_dir / "file_inventory.csv", index=False)

    outcome_path = choose_outcome_file(files, scope=scope)
    if outcome_path is None:
        raise FileNotFoundError("No mortality/incidence file found in repository.")

    if not scope:
        scope = infer_scope_from_path(outcome_path)
        if scope == "national":
            try:
                outcome_temp = load_tabular(outcome_path)
                outcome_cols = infer_geo_columns(outcome_temp)
                if outcome_cols.get("cve_ent"):
                    state_code = extract_digits(outcome_temp[outcome_cols["cve_ent"]]).dropna().iloc[0]
                    scope = {"24": "slp", "31": "merida"}.get(str(state_code).zfill(2), "national")
            except Exception:
                pass

    poly_path = choose_polygon_file(files, scope=scope)
    if poly_path is None:
        raise FileNotFoundError("No spatial polygon file found in repository.")
    inferred_scope = scope or infer_scope_from_path(poly_path)

    population_path = choose_population_file(files, scope=inferred_scope)
    exposure_paths = choose_exposure_files(files, scope=inferred_scope)
    if population_path is None:
        raise FileNotFoundError("No population file found in repository.")

    polygons = load_spatial(poly_path)
    poly_cols = infer_geo_columns(polygons)
    polygons["geo_id"] = build_geo_key(
        polygons,
        state_col=poly_cols.get("state"),
        municipio_col=poly_cols.get("municipio"),
        cve_ent_col=poly_cols.get("cve_ent"),
        cve_mun_col=poly_cols.get("cve_mun"),
    )
    polygons = polygons.dropna(subset=["geo_id"]).drop_duplicates(subset=["geo_id"])

    pop, pop_meta = build_population_table(population_path, scope=inferred_scope, age_adjustment=age_adjustment)
    pop_meta["population_file"] = _metadata_path(population_path, repo_root)
    outcome_agg, outcome_meta, outcome_audit = _build_outcome_table(outcome_path)
    outcome_meta["outcome_file"] = _metadata_path(outcome_path, repo_root)
    outcome_audit.to_csv(output_dir / "outcome_audit_raw.csv", index=False)

    years = sorted(outcome_agg["year"].dropna().unique().tolist()) if "year" in outcome_agg.columns else None
    analytic = _base_geography_frame(polygons, pop, years=years)
    join_cols = ["geo_id"] + (["year"] if years else [])
    analytic = analytic.merge(outcome_agg, on=join_cols, how="left")
    analytic["outcome_missing_flag"] = analytic["outcome_count"].isna().astype(int)

    population_warnings = validate_population(analytic, scope=inferred_scope, denominator_type=pop_meta.get("population_denominator_type", "total_population"))

    polygon_geo_ids = set(polygons["geo_id"].astype(str))
    outcome_geo_ids = set(outcome_agg["geo_id"].astype(str))
    missing_outcome = analytic.loc[analytic["outcome_missing_flag"] == 1, ["geo_id", "population"]].copy()
    missing_outcome.to_csv(output_dir / "missing_outcome_geos.csv", index=False)
    unmatched_outcomes = outcome_agg.loc[~outcome_agg["geo_id"].astype(str).isin(polygon_geo_ids)].copy()
    unmatched_outcomes.to_csv(output_dir / "unmatched_outcome_rows.csv", index=False)

    feature_tables = []
    exposure_meta = []
    for path in exposure_paths:
        try:
            df = load_tabular(path)
        except Exception as exc:
            exposure_meta.append({"file": _metadata_path(path, repo_root), "used": False, "reason": f"load_failed: {exc}"})
            continue
        cols = infer_geo_columns(df)
        if cols["longitude"] and cols["latitude"]:
            try:
                pts = points_from_tabular(df, cols["longitude"], cols["latitude"])
                source_name = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
                feature_tables.append(
                    compute_point_source_exposure(
                        polygons[["geo_id", "geometry"]].copy(),
                        pts,
                        area_key="geo_id",
                        config=ExposureConfig(source_name=source_name),
                    )
                )
                exposure_meta.append({"file": _metadata_path(path, repo_root), "used": True, "source_name": source_name, "n_points": int(len(pts))})
            except Exception as exc:
                exposure_meta.append({"file": _metadata_path(path, repo_root), "used": False, "reason": f"point_conversion_failed: {exc}"})
        else:
            exposure_meta.append({"file": _metadata_path(path, repo_root), "used": False, "reason": "no_lat_lon_columns_detected"})

    for ft in feature_tables:
        analytic = analytic.merge(ft, on="geo_id", how="left")

    protected = {"geo_id", "geometry", "outcome_count", "population", "year", "outcome_missing_flag", *pop_meta.get("age_adjustment_columns", [])}
    exposure_cols = [c for c in analytic.columns if c not in protected]
    for c in exposure_cols:
        analytic[c] = pd.to_numeric(analytic[c], errors="coerce").fillna(0)

    analytic["observed_rate_per_100k"] = analytic["outcome_count"] / analytic["population"] * 100000.0

    analytic_gdf = gpd.GeoDataFrame(analytic, geometry="geometry", crs=polygons.crs)
    analytic_gdf.to_file(output_dir / "analytic_dataset.geojson", driver="GeoJSON")
    pd.DataFrame(analytic_gdf.drop(columns="geometry")).to_csv(output_dir / "analytic_dataset.csv", index=False)

    metadata: Dict[str, object] = {
        "scope": inferred_scope,
        "polygon_file": _metadata_path(poly_path, repo_root),
        "n_polygon_areas": int(polygons["geo_id"].nunique()),
        "n_rows_after_all_polygon_retention": int(len(analytic_gdf)),
        "n_missing_outcome_rows": int(analytic_gdf["outcome_missing_flag"].sum()),
        "missing_outcome_geo_ids": sorted(analytic_gdf.loc[analytic_gdf["outcome_missing_flag"] == 1, "geo_id"].astype(str).tolist()),
        "unmatched_outcome_geo_count": int(len(unmatched_outcomes)),
        "population_warnings": population_warnings,
        **pop_meta,
        **outcome_meta,
        "exposure_files": exposure_meta,
    }
    return analytic_gdf, metadata


def _write_feature_transform(feature_stats: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    rows = []
    for feature, stats in feature_stats.items():
        row = {"feature": feature, **stats}
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "feature_standardization.csv", index=False)


def _standardize_control_columns(df: pd.DataFrame, control_cols: List[str], output_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Standardize non-exposure controls and drop exact duplicates.

    Duplicate controls can arise when helper features such as an "older priority"
    age share copy the same raw column as an already included age share. Keeping
    both creates a rank-deficient design matrix and can make single-family
    sensitivity models fall back to fixed-alpha GLM fits.
    """
    out = df.copy()
    z_cols: List[str] = []
    rows = []
    seen_signatures = {}

    for c in control_cols:
        if c not in out.columns:
            rows.append({"control": c, "standardized_control": "", "status": "drop", "reason": "missing_column"})
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        if s.notna().sum() == 0:
            rows.append({"control": c, "standardized_control": "", "status": "drop", "reason": "all_missing"})
            continue
        if s.nunique(dropna=True) <= 1:
            rows.append({"control": c, "standardized_control": "", "status": "drop", "reason": "constant"})
            continue

        fill = float(s.median(skipna=True))
        filled = s.fillna(fill).astype(float)
        signature = tuple(np.round(filled.to_numpy(), 12))
        if signature in seen_signatures:
            rows.append({
                "control": c,
                "standardized_control": "",
                "status": "drop",
                "reason": f"exact_duplicate_of_{seen_signatures[signature]}",
                "fill_value": fill,
                "mean": float(filled.mean()),
                "std": float(filled.std(ddof=0)),
            })
            continue
        seen_signatures[signature] = c

        mean = float(filled.mean())
        std = float(filled.std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            rows.append({"control": c, "standardized_control": "", "status": "drop", "reason": "zero_std"})
            continue
        z = f"control_{c}_z"
        out[z] = (filled - mean) / std
        z_cols.append(z)
        rows.append({
            "control": c,
            "standardized_control": z,
            "status": "keep",
            "reason": "passes_filters",
            "fill_value": fill,
            "mean": mean,
            "std": std,
        })

    pd.DataFrame(rows).to_csv(output_dir / "age_adjustment_report.csv", index=False)
    return out, z_cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SANA breast cancer risk baseline pipeline.")
    parser.add_argument("--repo-root", type=str, required=True, help="Path to the cloned repository root.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for outputs.")
    parser.add_argument("--scope", type=str, choices=["slp", "merida", "national"], default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bootstrap-draws", type=int, default=100)
    parser.add_argument("--min-nonzero-feature", type=int, default=5)
    parser.add_argument("--corr-threshold", type=float, default=0.98)
    parser.add_argument("--age-adjustment", choices=["auto", "off"], default="auto")
    parser.add_argument(
        "--missing-outcome-policy",
        choices=["exclude", "zero_fill"],
        default="exclude",
        help="Default is exclude: keep missing-outcome geographies in maps, but exclude them from the likelihood.",
    )
    parser.add_argument("--skip-sensitivity", action="store_true", help="Skip single-exposure-family sensitivity models.")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analytic, metadata = build_analytic_dataset(repo_root, output_dir, scope=args.scope, age_adjustment=args.age_adjustment)
    base_df = pd.DataFrame(analytic.drop(columns="geometry")).copy()

    # Build areal adjacency outputs early. These are used by the Bayesian spatial
    # hierarchical model (ICAR/CAR/BYM-style random effects). The adjacency matrix
    # is aligned to the retained polygon/order of the analytic dataset.
    adjacency_outputs = write_adjacency_outputs(
        analytic[["geo_id", "geometry"]].drop_duplicates(subset=["geo_id"]),
        output_dir=output_dir,
        geo_id_col="geo_id",
        connect_isolates=True,
        connect_components=True,
    )

    age_raw_cols = metadata.get("age_adjustment_columns", []) if args.age_adjustment == "auto" else []
    base_df, age_control_cols = _standardize_control_columns(base_df, list(age_raw_cols), output_dir)
    if not age_control_cols and not (output_dir / "age_adjustment_report.csv").exists():
        pd.DataFrame([{"status": "no_age_adjustment_columns_available"}]).to_csv(output_dir / "age_adjustment_report.csv", index=False)

    excluded = {"geo_id", "outcome_count", "population", "year", "outcome_missing_flag", "observed_rate_per_100k", *age_raw_cols, *age_control_cols}
    candidate_features = [c for c in base_df.columns if c not in excluded]
    feature_report = diagnose_feature_columns(
        base_df,
        candidate_features,
        min_nonzero=args.min_nonzero_feature,
        corr_threshold=args.corr_threshold,
    )
    feature_report.to_csv(output_dir / "feature_filter_report.csv", index=False)
    feature_cols = feature_report.loc[feature_report["status"] == "keep", "feature"].tolist()

    if not feature_cols:
        raise ValueError("No valid exposure feature columns remain after stricter filtering.")

    base_df["model_inclusion_flag"] = 1
    # This column is used both for fitting and prediction. Missing-outcome rows are
    # excluded from the likelihood under the default policy, but they still need a
    # finite placeholder so the prediction routine can produce covariate-based maps.
    base_df["outcome_count_for_model"] = base_df["outcome_count"].fillna(0)
    if args.missing_outcome_policy == "exclude":
        base_df.loc[base_df["outcome_missing_flag"] == 1, "model_inclusion_flag"] = 0

    model_df = base_df.loc[base_df["model_inclusion_flag"] == 1].copy()

    res = fit_negative_binomial(
        model_df,
        outcome_col="outcome_count_for_model",
        population_col="population",
        feature_cols=feature_cols,
        control_cols=age_control_cols,
    )
    res.summary_table.to_csv(output_dir / "model_summary.csv", index=False)
    _write_feature_transform(res.feature_stats, output_dir)

    preds_all = predict_counts(
        res,
        base_df,
        outcome_col="outcome_count_for_model",
        population_col="population",
        control_cols=age_control_cols,
    ).set_index("row_index")
    base_df.loc[preds_all.index, "predicted_count"] = preds_all["predicted_count"].values
    base_df.loc[preds_all.index, "predicted_rate_per_100k"] = preds_all["predicted_rate_per_100k"].values
    base_df.to_csv(output_dir / "modeled_dataset.csv", index=False)

    stability = bootstrap_ranking_stability(
        model_df,
        geo_id_col="geo_id",
        outcome_col="outcome_count_for_model",
        population_col="population",
        feature_cols=feature_cols,
        control_cols=age_control_cols,
        top_k=args.top_k,
        n_boot=args.bootstrap_draws,
        prediction_df=base_df,
    )
    stability.to_csv(output_dir / "ranking_stability.csv", index=False)

    if not args.skip_sensitivity:
        sens_summary, sens_metrics, sens_rankings = run_single_family_sensitivity(
            training_df=model_df,
            prediction_df=base_df,
            geo_id_col="geo_id",
            outcome_col="outcome_count_for_model",
            population_col="population",
            feature_cols=feature_cols,
            control_cols=age_control_cols,
            top_k=args.top_k,
        )
        sens_summary.to_csv(output_dir / "sensitivity_single_family_model_summary.csv", index=False)
        sens_metrics.to_csv(output_dir / "sensitivity_single_family_metrics.csv", index=False)
        sens_rankings.to_csv(output_dir / "sensitivity_single_family_rankings.csv", index=False)

    write_bayesian_nb_inputs(
        df=base_df,
        output_dir=output_dir,
        geo_id_col="geo_id",
        outcome_col="outcome_count_for_model",
        population_col="population",
        feature_cols=feature_cols,
        control_cols=age_control_cols,
        inclusion_col="model_inclusion_flag",
    )
    write_bayesian_spatial_config(
        output_dir=output_dir,
        geo_id_col="geo_id",
        outcome_col="outcome_count_for_model",
        population_col="population",
        feature_cols=feature_cols,
        control_cols=age_control_cols,
        inclusion_col="model_inclusion_flag",
        bayesian_input_csv="bayesian_nb_input.csv",
        adjacency_matrix_csv="spatial_adjacency_matrix.csv",
        adjacency_metadata_json="spatial_adjacency_metadata.json",
    )

    risk_cube = build_risk_cube(
        base_df,
        geo_id_col="geo_id",
        year_col="year" if "year" in base_df.columns else None,
        predicted_rate_col="predicted_rate_per_100k",
        predicted_count_col="predicted_count",
        observed_rate_col="observed_rate_per_100k",
        source_label=res.model_type,
    )
    risk_cube.to_csv(output_dir / "risk_cube.csv", index=False)

    map_df = base_df.merge(stability, on="geo_id", how="left")
    geo_out = export_geo_outputs(
        analytic[["geo_id", "geometry"]].drop_duplicates(subset=["geo_id"]),
        map_df,
        geo_id_col="geo_id",
        out_dir=output_dir,
        geojson_name="risk_map.geojson",
    )

    metadata.update(
        {
            "missing_outcome_policy": args.missing_outcome_policy,
            "top_k": int(args.top_k),
            "bootstrap_draws": int(args.bootstrap_draws),
            "n_rows_used_in_likelihood": int(len(model_df)),
            "n_rows_predicted": int(len(base_df)),
            "n_rows_modeled": int(len(base_df)),
            "n_features_candidate": int(len(candidate_features)),
            "n_features_used": int(len(feature_cols)),
            "feature_columns_used": feature_cols,
            "age_control_columns_used": age_control_cols,
            "min_nonzero_feature": int(args.min_nonzero_feature),
            "corr_threshold": float(args.corr_threshold),
            "population_min": float(base_df["population"].min()),
            "population_max": float(base_df["population"].max()),
            "population_total_modeled": float(base_df[["geo_id", "population"]].drop_duplicates()["population"].sum()),
            "observed_rate_min_per_100k": float(base_df["observed_rate_per_100k"].min(skipna=True)),
            "observed_rate_max_per_100k": float(base_df["observed_rate_per_100k"].max(skipna=True)),
            "predicted_rate_min_per_100k": float(base_df["predicted_rate_per_100k"].min()),
            "predicted_rate_max_per_100k": float(base_df["predicted_rate_per_100k"].max()),
            "model_type": res.model_type,
            "model_fit_warnings": res.fit_warnings,
            "successful_bootstrap_fits": int(stability["successful_bootstrap_fits"].max()) if not stability.empty else 0,
            "requested_bootstrap_fits": int(args.bootstrap_draws),
            "risk_map_geojson": _metadata_path(geo_out, output_dir),
            "single_family_sensitivity_outputs": [] if args.skip_sensitivity else [
                "sensitivity_single_family_model_summary.csv",
                "sensitivity_single_family_metrics.csv",
                "sensitivity_single_family_rankings.csv",
            ],
            "bayesian_nb_inputs": ["bayesian_nb_input.csv", "bayesian_model_config.json"],
            "spatial_adjacency_outputs": {
                "matrix": "spatial_adjacency_matrix.csv",
                "neighbors": "spatial_neighbors.csv",
                "metadata": "spatial_adjacency_metadata.json",
            },
            "spatial_adjacency_summary": adjacency_outputs.get("metadata_dict", {}),
            "bayesian_spatial_inputs": ["bayesian_nb_input.csv", "bayesian_spatial_model_config.json", "spatial_adjacency_matrix.csv"],
        }
    )
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Wrote outputs to: {output_dir}")
    print(f"GeoJSON map: {geo_out}")
    print(f"Model type: {res.model_type}")
    if res.fit_warnings:
        print("Warnings:")
        for warning in res.fit_warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
