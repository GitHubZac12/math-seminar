
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


@dataclass
class ExposureConfig:
    source_name: str
    weight_col: Optional[str] = None
    radii_m: Sequence[float] = (1000.0, 5000.0, 10000.0)
    kernel_scale_m: float = 5000.0
    kernel_type: str = "inverse_distance"  # inverse_distance | gaussian
    eps_m: float = 100.0


def to_projected(gdf: gpd.GeoDataFrame, epsg: int = 3857) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs(epsg=epsg)


def points_from_tabular(
    df: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    missing = [c for c in [lon_col, lat_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing coordinate columns: {missing}")
    out = df.copy()
    out = out.dropna(subset=[lon_col, lat_col]).copy()
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out = out.dropna(subset=[lon_col, lat_col]).copy()
    gdf = gpd.GeoDataFrame(out, geometry=gpd.points_from_xy(out[lon_col], out[lat_col]), crs=crs)
    return gdf


def polygon_centroids(polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    projected = to_projected(polygons)
    cent = projected.copy()
    cent["geometry"] = cent.geometry.centroid
    return cent


def _kernel(distance_m: np.ndarray, config: ExposureConfig) -> np.ndarray:
    if config.kernel_type == "gaussian":
        return np.exp(-(distance_m ** 2) / (2 * config.kernel_scale_m ** 2))
    return 1.0 / (distance_m + config.eps_m)


def compute_point_source_exposure(
    areas: gpd.GeoDataFrame,
    sources: gpd.GeoDataFrame,
    area_key: str,
    config: ExposureConfig,
) -> pd.DataFrame:
    """
    Build area-level exposure features from point sources.
    Features:
    - kernel intensity
    - nearest source distance
    - counts within multiple radii
    """
    if area_key not in areas.columns:
        raise KeyError(f"`{area_key}` not found in area polygons")

    areas_p = to_projected(areas)
    centroids = polygon_centroids(areas)
    sources_p = to_projected(sources)

    area_ids = centroids[area_key].astype(str).tolist()
    cent_geom = np.array([(g.x, g.y) for g in centroids.geometry])
    src_geom = np.array([(g.x, g.y) for g in sources_p.geometry])

    if len(src_geom) == 0:
        out = pd.DataFrame({area_key: area_ids})
        out[f"{config.source_name}_kernel_intensity"] = 0.0
        out[f"{config.source_name}_nearest_source_m"] = np.nan
        for r in config.radii_m:
            out[f"{config.source_name}_count_within_{int(r)}m"] = 0
        return out

    dx = cent_geom[:, None, 0] - src_geom[None, :, 0]
    dy = cent_geom[:, None, 1] - src_geom[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)

    if config.weight_col and config.weight_col in sources_p.columns:
        weights = pd.to_numeric(sources_p[config.weight_col], errors="coerce").fillna(1.0).to_numpy()
    else:
        weights = np.ones(len(sources_p), dtype=float)

    kernel_vals = _kernel(dist, config)
    kernel_intensity = (kernel_vals * weights[None, :]).sum(axis=1)
    nearest = dist.min(axis=1)

    out = pd.DataFrame({area_key: area_ids})
    out[f"{config.source_name}_kernel_intensity"] = kernel_intensity
    out[f"{config.source_name}_nearest_source_m"] = nearest

    for r in config.radii_m:
        out[f"{config.source_name}_count_within_{int(r)}m"] = (dist <= r).sum(axis=1)

    return out


def aggregate_points_within_polygons(
    areas: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    area_key: str,
    value_cols: Sequence[str],
    prefix: str,
) -> pd.DataFrame:
    """Spatially join points into polygons and compute summary statistics."""
    areas = areas.copy()
    points = points.copy()
    areas = to_projected(areas)
    points = to_projected(points)

    joined = gpd.sjoin(points, areas[[area_key, "geometry"]], how="left", predicate="within")
    grouped = joined.groupby(area_key, dropna=True)

    frames = []
    for col in value_cols:
        if col not in joined.columns:
            continue
        s = pd.to_numeric(joined[col], errors="coerce")
        temp = pd.DataFrame(
            {
                area_key: joined[area_key],
                f"{prefix}_{col}_mean": s,
                f"{prefix}_{col}_median": s,
                f"{prefix}_{col}_max": s,
            }
        )
        summary = temp.groupby(area_key).agg(
            {
                f"{prefix}_{col}_mean": "mean",
                f"{prefix}_{col}_median": "median",
                f"{prefix}_{col}_max": "max",
            }
        )
        frames.append(summary)

    if not frames:
        return pd.DataFrame({area_key: areas[area_key].astype(str)}).drop_duplicates()

    out = pd.concat(frames, axis=1).reset_index()
    return out


def merge_feature_tables(area_key: str, *tables: pd.DataFrame) -> pd.DataFrame:
    if not tables:
        raise ValueError("At least one table is required")
    out = tables[0].copy()
    for t in tables[1:]:
        out = out.merge(t, on=area_key, how="outer")
    return out
