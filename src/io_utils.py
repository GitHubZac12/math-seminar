from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Sequence

import geopandas as gpd
import pandas as pd


EXCEL_SUFFIXES = {".xlsx", ".xls", ".xlsm"}
TABULAR_SUFFIXES = EXCEL_SUFFIXES | {".csv"}
SPATIAL_SUFFIXES = {".shp", ".geojson", ".gpkg", ".json"}
SKIP_DIR_NAMES = {".git", "__pycache__", ".ipynb_checkpoints", "outputs", "output", "tmp", "temp"}


def discover_files(repo_root: Path) -> pd.DataFrame:
    """Recursively discover candidate files in the repository.

    The discovery step intentionally skips common generated-output directories so that
    files produced by previous pipeline runs are not accidentally re-ingested as inputs.
    """
    repo_root = Path(repo_root)
    rows = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part.lower() in SKIP_DIR_NAMES for part in path.parts):
            continue
        suffix = path.suffix.lower()
        if suffix in TABULAR_SUFFIXES or suffix in SPATIAL_SUFFIXES:
            rows.append(
                {
                    "path": str(path),
                    "name": path.name,
                    "suffix": suffix,
                    "parent": str(path.parent),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["path", "name", "suffix", "parent"])
    return pd.DataFrame(rows).sort_values(["parent", "name"]).reset_index(drop=True)


def normalize_name(name: str) -> str:
    """Normalize a string for robust column matching."""
    text = str(name).strip().lower()
    text = text.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    text = text.replace("ñ", "n")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_name(c) for c in out.columns]
    return out


def first_matching_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = list(df.columns)
    norm_cols = {normalize_name(c): c for c in cols}
    for cand in candidates:
        n = normalize_name(cand)
        if n in norm_cols:
            return norm_cols[n]
    for col in cols:
        ncol = normalize_name(col)
        for cand in candidates:
            ncand = normalize_name(cand)
            if ncand in ncol or ncol in ncand:
                return col
    return None


def infer_geo_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    candidates = {
        "state": ["estado", "nom_ent", "entidad", "state", "nombre_entidad", "nomgeo", "nombre_estado"],
        "municipio": [
            "municipio",
            "nom_mun",
            "nombre_municipio",
            "mun_name",
            "mpio",
            "municipality",
            "nomgeo",
        ],
        "locality_name": ["nom_loc", "nombre_localidad", "localidad", "loc_name"],
        "cve_ent": ["cve_ent", "ent", "state_code", "clave_entidad", "entidad_code"],
        "cve_mun": ["cve_mun", "mun", "municipio_code", "clave_municipio", "muni", "mpio_code"],
        "cve_loc": ["cve_loc", "loc", "loc_code", "localidad_code", "clave_localidad"],
        "year": ["anio", "ano", "year", "periodo", "ejercicio"],
        "population": [
            "pobfem",
            "pob_fem",
            "pob_mujeres",
            "mujeres",
            "female_population",
            "pobtot",
            "pob_total",
            "poblacion_total",
            "poblacion",
            "population",
            "tot_pob",
            "p_total",
            "total",
        ],
        "incidence": [
            "incidencia",
            "casos",
            "cases",
            "incident_cases",
            "n_casos",
        ],
        "mortality": [
            "mortalidad",
            "muertes",
            "deaths",
            "defunciones",
            "n_muertes",
        ],
        "latitude": ["lat", "latitude", "latitud", "y", "coord_y", "lat_decimal"],
        "longitude": ["lon", "lng", "long", "longitude", "longitud", "x", "coord_x", "lon_decimal"],
    }
    return {k: first_matching_column(df, v) for k, v in candidates.items()}


def _score_excel_sheet(sheet_name: str, sample: pd.DataFrame) -> int:
    name = normalize_name(sheet_name)
    score = 0
    if any(bad in name for bad in ["nota", "notas", "metadata", "metadatos", "catalogo", "indice"]):
        score -= 50
    if any(good in name for good in ["datos", "data", "iter", "mortal", "pobl", "pob", "casos", "incid"]):
        score += 20
    score += min(len(sample), 20)
    score += min(len(sample.columns), 30)
    normalized_cols = {normalize_name(c) for c in sample.columns}
    for important in ["cve_ent", "cve_mun", "cve_loc", "pobtot", "pobfem", "defunciones", "casos"]:
        if important in normalized_cols:
            score += 25
    return score


def load_tabular(path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return standardize_columns(pd.read_csv(path))
    if path.suffix.lower() in EXCEL_SUFFIXES:
        if sheet_name is None:
            xls = pd.ExcelFile(path)
            scored = []
            for sheet in xls.sheet_names:
                try:
                    sample = pd.read_excel(path, sheet_name=sheet, nrows=25)
                    scored.append((_score_excel_sheet(sheet, sample), sheet))
                except Exception:
                    scored.append((-999, sheet))
            sheet_name = sorted(scored, reverse=True)[0][1]
        return standardize_columns(pd.read_excel(path, sheet_name=sheet_name))
    raise ValueError(f"Unsupported tabular file type: {path}")


def load_spatial(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    gdf = gdf.rename(columns={c: normalize_name(c) for c in gdf.columns})
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def extract_digits(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)", expand=False)


def build_geo_key(
    df: pd.DataFrame,
    state_col: Optional[str] = None,
    municipio_col: Optional[str] = None,
    cve_ent_col: Optional[str] = None,
    cve_mun_col: Optional[str] = None,
) -> pd.Series:
    """Create a robust 5-character municipality key when possible.

    Preferred key: two-digit entity code + three-digit municipality code.
    Fallback key: normalized state and municipality names.
    """
    if "cvegeo" in df.columns:
        digits = extract_digits(df["cvegeo"])
        return digits.str[:5].str.zfill(5)

    if cve_ent_col and cve_mun_col and cve_ent_col in df.columns and cve_mun_col in df.columns:
        ent = extract_digits(df[cve_ent_col]).str.zfill(2)
        mun = extract_digits(df[cve_mun_col]).str.zfill(3)
        return ent + mun

    state = (
        df[state_col].astype(str).str.strip().str.lower().map(normalize_name)
        if state_col and state_col in df.columns else ""
    )
    mun = (
        df[municipio_col].astype(str).str.strip().str.lower().map(normalize_name)
        if municipio_col and municipio_col in df.columns else ""
    )
    result = state + "::" + mun
    if isinstance(result, pd.Series):
        return result.str.strip(":")
    return pd.Series([result.strip(":")] * len(df), index=df.index)


def safe_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("\u00a0", "", regex=False)
        .replace({"nan": None, "None": None, "": None, "*": None, "N/D": None, "ND": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def choose_population_column(df: pd.DataFrame) -> Optional[str]:
    """Choose a denominator column, preferring female population when available.

    Breast-cancer outcomes should generally use a female-population denominator if the
    source provides one. If not, the function falls back to total population.
    """
    excluded = set()
    cols = infer_geo_columns(df)
    for key in ["cve_ent", "cve_mun", "cve_loc", "year", "incidence", "mortality"]:
        if cols.get(key):
            excluded.add(cols[key])

    priority_patterns = [
        (r"^(pobfem|pob_fem|pob_mujeres|mujeres|female_population)$", 10000),
        (r"pob.*fem|fem|mujer", 8000),
        (r"^(pobtot|pob_total|poblacion_total|population|poblacion)$", 5000),
        (r"pobtot|poblacion|population|tot_pob|p_total", 4000),
        (r"total", 1000),
    ]

    candidates = []
    for col in df.columns:
        if col in excluded:
            continue
        s = safe_numeric(df[col])
        non_null = int(s.notna().sum())
        if non_null == 0:
            continue
        max_val = s.max(skipna=True)
        if pd.isna(max_val) or max_val <= 0:
            continue
        score = non_null
        for pat, bonus in priority_patterns:
            if re.search(pat, normalize_name(col)):
                score += bonus
        # Demote age-bin variables; they are denominators for subgroups, not full municipal denominators.
        if re.search(r"pob\d|edad|age|quinquen", normalize_name(col)):
            score -= 3000
        candidates.append((score, col))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]
