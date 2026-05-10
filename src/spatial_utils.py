from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd


def _connected_components(W: np.ndarray) -> List[List[int]]:
    n = W.shape[0]
    visited = np.zeros(n, dtype=bool)
    components: List[List[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        q: deque[int] = deque([start])
        visited[start] = True
        comp: List[int] = []
        while q:
            i = q.popleft()
            comp.append(i)
            for j in np.flatnonzero(W[i] > 0):
                if not visited[j]:
                    visited[j] = True
                    q.append(int(j))
        components.append(comp)
    return components


def build_contiguity_adjacency(
    areas: gpd.GeoDataFrame,
    geo_id_col: str = "geo_id",
    projected_epsg: int = 3857,
    connect_isolates: bool = True,
    connect_components: bool = True,
) -> tuple[list[str], np.ndarray, pd.DataFrame, dict]:
    """Build a queen-contiguity adjacency matrix from polygon geometries.

    The resulting matrix is appropriate for areal spatial priors such as ICAR/CAR.
    If a municipality has no detected neighbors because of polygon precision issues,
    it can be connected to its nearest centroid neighbor. Disconnected components can
    also be linked by nearest centroids so PyMC's ICAR prior has a single connected
    graph by default.
    """
    if geo_id_col not in areas.columns:
        raise KeyError(f"{geo_id_col} is not present in the area GeoDataFrame")
    if "geometry" not in areas.columns:
        raise KeyError("Input must contain a geometry column")

    gdf = areas[[geo_id_col, "geometry"]].drop_duplicates(subset=[geo_id_col]).copy()
    gdf[geo_id_col] = gdf[geo_id_col].astype(str)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(epsg=projected_epsg)
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf.reset_index(drop=True)

    geo_ids = gdf[geo_id_col].astype(str).tolist()
    n = len(gdf)
    W = np.zeros((n, n), dtype=int)

    sindex = gdf.sindex
    for i, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            continue
        candidate_idx = list(sindex.query(geom, predicate="intersects"))
        for j in candidate_idx:
            j = int(j)
            if j <= i:
                continue
            other = gdf.geometry.iloc[j]
            if other is None or other.is_empty:
                continue
            # Queen contiguity: sharing a point, edge, or tiny overlap all count.
            if geom.touches(other) or geom.intersects(other):
                W[i, j] = 1
                W[j, i] = 1

    centroids = np.array([(geom.centroid.x, geom.centroid.y) for geom in gdf.geometry])
    degrees_before = W.sum(axis=1)
    isolates_before = [geo_ids[i] for i in np.where(degrees_before == 0)[0]]
    artificial_edges: list[dict] = []

    def add_nearest_edge(i: int, candidate_pool: Optional[Sequence[int]] = None, reason: str = "nearest_neighbor") -> None:
        if candidate_pool is None:
            candidate_pool = [j for j in range(n) if j != i]
        candidate_pool = [j for j in candidate_pool if j != i]
        if not candidate_pool:
            return
        diffs = centroids[np.array(candidate_pool)] - centroids[i]
        distances = np.sqrt((diffs * diffs).sum(axis=1))
        j = int(candidate_pool[int(np.argmin(distances))])
        W[i, j] = 1
        W[j, i] = 1
        artificial_edges.append(
            {
                "geo_id_1": geo_ids[i],
                "geo_id_2": geo_ids[j],
                "reason": reason,
                "centroid_distance_m": float(np.min(distances)),
            }
        )

    if connect_isolates:
        for i in np.where(W.sum(axis=1) == 0)[0]:
            add_nearest_edge(int(i), reason="isolate_nearest_neighbor")

    components_before_bridge = _connected_components(W)
    if connect_components and len(components_before_bridge) > 1:
        # Greedily connect each smaller component to the first component using nearest centroids.
        base = components_before_bridge[0]
        for comp in components_before_bridge[1:]:
            best_pair = None
            best_dist = np.inf
            for i in comp:
                diffs = centroids[np.array(base)] - centroids[i]
                distances = np.sqrt((diffs * diffs).sum(axis=1))
                k = int(np.argmin(distances))
                if float(distances[k]) < best_dist:
                    best_dist = float(distances[k])
                    best_pair = (i, int(base[k]))
            if best_pair is not None:
                i, j = best_pair
                W[i, j] = 1
                W[j, i] = 1
                artificial_edges.append(
                    {
                        "geo_id_1": geo_ids[i],
                        "geo_id_2": geo_ids[j],
                        "reason": "component_bridge_nearest_neighbor",
                        "centroid_distance_m": best_dist,
                    }
                )
                base = base + comp

    degrees_after = W.sum(axis=1)
    components_after = _connected_components(W)

    edge_rows = []
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] == 1:
                diffs = centroids[i] - centroids[j]
                edge_rows.append(
                    {
                        "geo_id_1": geo_ids[i],
                        "geo_id_2": geo_ids[j],
                        "centroid_distance_m": float(np.sqrt(np.dot(diffs, diffs))),
                        "artificial_edge": False,
                        "reason": "queen_contiguity",
                    }
                )
    artificial_lookup = {tuple(sorted((e["geo_id_1"], e["geo_id_2"]))): e for e in artificial_edges}
    for row in edge_rows:
        key = tuple(sorted((row["geo_id_1"], row["geo_id_2"])))
        if key in artificial_lookup:
            row["artificial_edge"] = True
            row["reason"] = artificial_lookup[key]["reason"]
            row["centroid_distance_m"] = artificial_lookup[key]["centroid_distance_m"]

    neighbors = pd.DataFrame(edge_rows)
    metadata = {
        "geo_id_col": geo_id_col,
        "n_nodes": int(n),
        "n_edges": int(W.sum() // 2),
        "is_symmetric": bool(np.all(W == W.T)),
        "diagonal_zero": bool(np.all(np.diag(W) == 0)),
        "min_neighbors": int(degrees_after.min()) if n else 0,
        "max_neighbors": int(degrees_after.max()) if n else 0,
        "mean_neighbors": float(degrees_after.mean()) if n else 0.0,
        "isolates_before_connection": isolates_before,
        "isolates_after_connection": [geo_ids[i] for i in np.where(degrees_after == 0)[0]],
        "n_components_after_connection": int(len(components_after)),
        "connect_isolates": bool(connect_isolates),
        "connect_components": bool(connect_components),
        "n_artificial_edges": int(len(artificial_edges)),
        "artificial_edges": artificial_edges,
        "projected_epsg": int(projected_epsg),
        "spatial_prior_recommendation": "Use ICAR/CAR/BYM-style areal random effects with this adjacency matrix.",
    }
    return geo_ids, W, neighbors, metadata


def write_adjacency_outputs(
    areas: gpd.GeoDataFrame,
    output_dir: Path,
    geo_id_col: str = "geo_id",
    projected_epsg: int = 3857,
    connect_isolates: bool = True,
    connect_components: bool = True,
) -> dict:
    """Build and write adjacency artifacts used by the spatial Bayesian model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geo_ids, W, neighbors, metadata = build_contiguity_adjacency(
        areas=areas,
        geo_id_col=geo_id_col,
        projected_epsg=projected_epsg,
        connect_isolates=connect_isolates,
        connect_components=connect_components,
    )
    matrix_df = pd.DataFrame(W, columns=geo_ids)
    matrix_df.insert(0, geo_id_col, geo_ids)
    matrix_path = output_dir / "spatial_adjacency_matrix.csv"
    neighbors_path = output_dir / "spatial_neighbors.csv"
    metadata_path = output_dir / "spatial_adjacency_metadata.json"
    matrix_df.to_csv(matrix_path, index=False)
    neighbors.to_csv(neighbors_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "matrix": matrix_path,
        "neighbors": neighbors_path,
        "metadata": metadata_path,
        "metadata_dict": metadata,
    }


def load_adjacency_matrix(matrix_path: Path, geo_ids: Sequence[str], geo_id_col: str = "geo_id") -> np.ndarray:
    """Load and align an adjacency matrix to a specific geo_id order."""
    matrix_path = Path(matrix_path)
    df = pd.read_csv(matrix_path, dtype={geo_id_col: str})
    if geo_id_col not in df.columns:
        raise KeyError(f"{matrix_path} is missing {geo_id_col}")
    df[geo_id_col] = df[geo_id_col].astype(str)
    geo_ids = [str(g) for g in geo_ids]
    missing_rows = sorted(set(geo_ids) - set(df[geo_id_col]))
    missing_cols = sorted(set(geo_ids) - set(map(str, df.columns)))
    if missing_rows or missing_cols:
        raise KeyError(f"Adjacency matrix is not aligned. missing_rows={missing_rows}, missing_cols={missing_cols}")
    aligned = df.set_index(geo_id_col).loc[geo_ids, geo_ids]
    W = aligned.to_numpy(dtype=int)
    if not np.all(W == W.T):
        raise ValueError("Adjacency matrix must be symmetric")
    if not np.all(np.diag(W) == 0):
        raise ValueError("Adjacency matrix diagonal must be zero")
    if (W.sum(axis=1) == 0).any():
        bad = [geo_ids[i] for i in np.where(W.sum(axis=1) == 0)[0]]
        raise ValueError(f"Adjacency matrix has isolated nodes: {bad}")
    return W
