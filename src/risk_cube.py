from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd


def build_risk_cube(
    analytic_df: pd.DataFrame,
    geo_id_col: str,
    year_col: Optional[str],
    predicted_rate_col: str,
    predicted_count_col: str,
    observed_rate_col: Optional[str] = None,
    source_label: str = "baseline_nb",
) -> pd.DataFrame:
    metrics = [predicted_rate_col, predicted_count_col]
    if observed_rate_col and observed_rate_col in analytic_df.columns:
        metrics.append(observed_rate_col)

    id_cols = [geo_id_col]
    if year_col and year_col in analytic_df.columns:
        id_cols.append(year_col)

    work = analytic_df[id_cols + metrics].copy()
    out = []
    for metric in metrics:
        temp = work[id_cols + [metric]].copy()
        temp = temp.rename(columns={metric: "value"})
        temp["metric"] = metric
        temp["source"] = source_label
        out.append(temp)
    return pd.concat(out, ignore_index=True)


def export_geo_outputs(
    polygons: gpd.GeoDataFrame,
    analytic_df: pd.DataFrame,
    geo_id_col: str,
    out_dir: Path,
    geojson_name: str = "risk_map.geojson",
) -> Path:
    """Merge one clean geometry table with one non-geometry result table.

    This avoids the duplicate `_x`/`_y` fields that occur when two already-enriched
    dataframes are merged together.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    poly = polygons[[geo_id_col, "geometry"]].drop_duplicates(subset=[geo_id_col]).copy()
    if "geometry" in analytic_df.columns:
        analytic_df = analytic_df.drop(columns=["geometry"])
    attrs = analytic_df.loc[:, ~analytic_df.columns.duplicated()].copy()
    merged = poly.merge(attrs, on=geo_id_col, how="left")
    out_path = out_dir / geojson_name
    merged.to_file(out_path, driver="GeoJSON")
    return out_path
