#!/usr/bin/env python3
"""
Map grey-relative CIE-LUV target definitions to displayable colors on the new screen.

This stage depends on:
- the historical grey-relative target displacements from stage 01
- the fitted XYZ -> RGB and RGB -> XYZ MLPs from stage 02

Outputs are written under:

    calibration/outputs/03_map_cieluv_targets_to_new_screen/
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

try:
    from .rgb_to_xyz_mlp import RGBToXYZMLP
    from .xyz_to_rgb_mlp import XYZToRGBMLP
except ImportError:  # pragma: no cover
    from rgb_to_xyz_mlp import RGBToXYZMLP
    from xyz_to_rgb_mlp import XYZToRGBMLP


DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parent / "outputs" / "03_map_cieluv_targets_to_new_screen"
)
DEFAULT_FORWARD_MODEL = (
    Path(__file__).resolve().parent
    / "outputs"
    / "02_fit_screen_model"
    / "data"
    / "screen_xyz_to_rgb_mlp_model.pth"
)
DEFAULT_INVERSE_MODEL = (
    Path(__file__).resolve().parent
    / "outputs"
    / "02_fit_screen_model"
    / "data"
    / "screen_rgb_to_xyz_mlp_model.pth"
)
DEFAULT_MEASURED_XYZ = (
    Path(__file__).resolve().parent
    / "outputs"
    / "02_fit_screen_model"
    / "data"
    / "combined_measured_xyz_clean.csv"
)
DEFAULT_TARGET_DISPLACEMENTS = (
    Path(__file__).resolve().parent
    / "outputs"
    / "01_establish_historical_cieluv_target"
    / "data"
    / "compact_LUV_displacenments.csv"
)

XYZ_COLUMNS = {"rep", "id", "r", "g", "b", "X", "Y", "Z"}


def normalize_column_lookup(columns: list[str]) -> dict[str, str]:
    return {str(col).strip().lower(): col for col in columns}


def resolve_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lookup = normalize_column_lookup(df.columns.tolist())
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]
    raise ValueError(f"Could not resolve any of columns {candidates} from {list(df.columns)}")


def apply_name_suffix(path: Path, name_suffix: str) -> Path:
    if not name_suffix:
        return path
    if path.suffix:
        return path.with_name(f"{path.stem}{name_suffix}{path.suffix}")
    return path.with_name(f"{path.name}{name_suffix}")


def xyz_to_cie_luv(xyz_values: np.ndarray, reference_xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz_values, dtype=float)
    reference = np.asarray(reference_xyz, dtype=float).reshape(3)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz_values must be an Nx3 array.")

    Xn, Yn, Zn = reference
    if Yn <= 0:
        raise ValueError("Reference Y must be > 0.")

    den_ref = Xn + 15.0 * Yn + 3.0 * Zn
    if den_ref <= 0.0:
        raise ValueError("Invalid white reference for CIE-LUV conversion.")

    u_n = 4.0 * Xn / den_ref
    v_n = 9.0 * Yn / den_ref

    X = xyz[:, 0]
    Y = xyz[:, 1]
    Z = xyz[:, 2]
    den = X + 15.0 * Y + 3.0 * Z
    u_prime = np.where(den > 1e-12, 4.0 * X / den, 0.0)
    v_prime = np.where(den > 1e-12, 9.0 * Y / den, 0.0)

    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    yr = Y / Yn
    L = np.where(yr > epsilon, 116.0 * np.cbrt(np.maximum(yr, 0.0)) - 16.0, kappa * yr)
    u = 13.0 * L * (u_prime - u_n)
    v = 13.0 * L * (v_prime - v_n)
    return np.column_stack([L, u, v])


def cie_luv_to_xyz(luv_values: np.ndarray, reference_xyz: np.ndarray) -> np.ndarray:
    luv = np.asarray(luv_values, dtype=float)
    reference = np.asarray(reference_xyz, dtype=float).reshape(3)
    if luv.ndim != 2 or luv.shape[1] != 3:
        raise ValueError("luv_values must be an Nx3 array.")

    Xn, Yn, Zn = reference
    if Yn <= 0:
        raise ValueError("Reference Y must be > 0.")

    den_ref = Xn + 15.0 * Yn + 3.0 * Zn
    if den_ref <= 0.0:
        raise ValueError("Invalid white reference for CIE-LUV conversion.")

    u_n = 4.0 * Xn / den_ref
    v_n = 9.0 * Yn / den_ref

    L = luv[:, 0]
    u = luv[:, 1]
    v = luv[:, 2]

    with np.errstate(divide="ignore", invalid="ignore"):
        u_prime = np.where(L > 1e-12, u / (13.0 * L) + u_n, u_n)
        v_prime = np.where(L > 1e-12, v / (13.0 * L) + v_n, v_n)

    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    fy = (L + 16.0) / 116.0
    yr = np.where(L > kappa * epsilon, fy**3, L / kappa)
    Y = Yr = Yn * yr

    denom = np.where(np.abs(v_prime) > 1e-12, v_prime, np.nan)
    X = 9.0 * Y * u_prime / (4.0 * denom)
    Z = Y * (12.0 - 3.0 * u_prime - 20.0 * v_prime) / (4.0 * denom)
    xyz = np.column_stack([X, Yr, Z])
    xyz = np.nan_to_num(xyz, nan=-1.0, posinf=-1.0, neginf=-1.0)
    return xyz


def load_measured_xyz_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = XYZ_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required XYZ columns: {sorted(missing)}")

    work = df.copy()
    for column in ["rep", "id", "r", "g", "b", "X", "Y", "Z"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["rep", "id", "X", "Y", "Z"]).copy()
    work["rep"] = work["rep"].astype(int)
    work["id"] = work["id"].astype(int)

    valid_xyz = np.isfinite(work[["X", "Y", "Z"]]).all(axis=1)
    valid_xyz &= ~(work[["X", "Y", "Z"]] == -1.0).any(axis=1)
    valid_rgb = np.isfinite(work[["r", "g", "b"]]).all(axis=1)
    valid_rgb &= (work[["r", "g", "b"]] >= 0.0).all(axis=1)
    valid_rgb &= (work[["r", "g", "b"]] <= 255.0).all(axis=1)
    work = work[valid_xyz & valid_rgb].copy().reset_index(drop=True)
    return work


def load_target_displacements(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    id_col = resolve_column(df, ["id"])
    delta_l_col = resolve_column(df, ["delta_L", "delta_l", "deltal"])
    delta_u_col = resolve_column(df, ["delta_U", "delta_u", "deltau"])
    delta_v_col = resolve_column(df, ["delta_V", "delta_v", "deltav"])
    description_col = None
    try:
        description_col = resolve_column(df, ["description", "label", "name"])
    except ValueError:
        description_col = None

    out = pd.DataFrame(
        {
            "base_color_id": pd.to_numeric(df[id_col], errors="coerce"),
            "delta_L": pd.to_numeric(df[delta_l_col], errors="coerce"),
            "delta_U": pd.to_numeric(df[delta_u_col], errors="coerce"),
            "delta_V": pd.to_numeric(df[delta_v_col], errors="coerce"),
        }
    )
    if description_col is not None:
        out["description"] = df[description_col].astype(str)
    else:
        out["description"] = out["base_color_id"].map(lambda x: f"color_{int(x)}" if pd.notna(x) else "color")

    out = out.dropna(subset=["base_color_id", "delta_L", "delta_U", "delta_V"]).copy()
    out["base_color_id"] = out["base_color_id"].astype(int)
    out["target_order"] = np.arange(len(out), dtype=int)
    return out.reset_index(drop=True)


def sample_rgb_cube(grid_size: int) -> np.ndarray:
    if int(grid_size) < 2:
        raise ValueError("grid_size must be >= 2.")
    axis = np.linspace(0.0, 255.0, int(grid_size), dtype=float)
    rr, gg, bb = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.column_stack([rr.ravel(), gg.ravel(), bb.ravel()])


def compute_gamut_samples(
    rgb_to_xyz_model: RGBToXYZMLP,
    white_xyz: np.ndarray,
    rgb_grid_size: int,
) -> pd.DataFrame:
    rgb_grid = sample_rgb_cube(rgb_grid_size)
    bundle = rgb_to_xyz_model.predict_xyz(rgb_grid, warn=False, clip_min_zero=True)
    xyz = bundle.xyz_float
    luv = xyz_to_cie_luv(xyz, white_xyz)
    df = pd.DataFrame(
        {
            "r": rgb_grid[:, 0],
            "g": rgb_grid[:, 1],
            "b": rgb_grid[:, 2],
            "X": xyz[:, 0],
            "Y": xyz[:, 1],
            "Z": xyz[:, 2],
            "L": luv[:, 0],
            "u": luv[:, 1],
            "v": luv[:, 2],
        }
    )
    df["pred_invalid_xyz"] = np.any(bundle.invalid_mask, axis=1)
    return df


def select_l_slice_bounds(
    gamut_df: pd.DataFrame,
    grey_L: float,
    min_points: int = 50,
) -> tuple[tuple[float, float, float, float], pd.DataFrame, float]:
    tolerance_steps = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    slice_df = pd.DataFrame()
    used_tol = tolerance_steps[-1]
    for tol in tolerance_steps:
        subset = gamut_df[np.abs(gamut_df["L"] - grey_L) <= tol].copy()
        if len(subset) >= min_points:
            slice_df = subset
            used_tol = tol
            break
        if len(subset) > len(slice_df):
            slice_df = subset
            used_tol = tol

    if slice_df.empty:
        slice_df = gamut_df.copy()

    u_min = float(slice_df["u"].min())
    u_max = float(slice_df["u"].max())
    v_min = float(slice_df["v"].min())
    v_max = float(slice_df["v"].max())

    u_margin = 0.06 * max(u_max - u_min, 1.0)
    v_margin = 0.06 * max(v_max - v_min, 1.0)
    bounds = (u_min - u_margin, u_max + u_margin, v_min - v_margin, v_max + v_margin)
    return bounds, slice_df.reset_index(drop=True), used_tol


def evaluate_luv_gamut_mask(
    xyz_to_rgb_model: XYZToRGBMLP,
    white_xyz: np.ndarray,
    level_L: float,
    u_values: np.ndarray,
    v_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uu, vv = np.meshgrid(u_values, v_values, indexing="xy")
    flat_luv = np.column_stack(
        [
            np.full(uu.size, float(level_L), dtype=float),
            uu.ravel(),
            vv.ravel(),
        ]
    )
    flat_xyz = cie_luv_to_xyz(flat_luv, white_xyz)
    valid_xyz = np.isfinite(flat_xyz).all(axis=1) & (flat_xyz >= 0.0).all(axis=1)

    bundle = xyz_to_rgb_model.predict_rgb(flat_xyz, warn=False, clip=False)
    raw_rgb = np.asarray(bundle.rgb_float, dtype=float)
    valid_rgb = np.isfinite(raw_rgb).all(axis=1)
    valid_rgb &= (raw_rgb >= 0.0).all(axis=1)
    valid_rgb &= (raw_rgb <= 255.0).all(axis=1)

    valid_mask = (valid_xyz & valid_rgb).reshape(len(v_values), len(u_values))
    return valid_mask, flat_xyz.reshape(len(v_values), len(u_values), 3), raw_rgb.reshape(len(v_values), len(u_values), 3)


def compute_valid_distance_map(valid_mask: np.ndarray, du: float, dv: float) -> np.ndarray:
    padded = np.pad(valid_mask.astype(bool), 1, constant_values=False)
    dist = distance_transform_edt(padded, sampling=(dv, du))
    return dist[1:-1, 1:-1]


def interpolate_grid_value(
    x_values: np.ndarray,
    y_values: np.ndarray,
    grid: np.ndarray,
    x: float,
    y: float,
) -> float:
    x_axis = np.asarray(x_values, dtype=float)
    y_axis = np.asarray(y_values, dtype=float)
    values = np.asarray(grid, dtype=float)
    if values.shape != (len(y_axis), len(x_axis)):
        raise ValueError("grid shape must be (len(y_values), len(x_values)).")
    if len(x_axis) == 0 or len(y_axis) == 0:
        raise ValueError("Grid axes must be non-empty.")
    if len(x_axis) == 1 or len(y_axis) == 1:
        return float(values[0, 0])

    x_clamped = float(np.clip(x, x_axis[0], x_axis[-1]))
    y_clamped = float(np.clip(y, y_axis[0], y_axis[-1]))

    x_hi = int(np.searchsorted(x_axis, x_clamped, side="right"))
    y_hi = int(np.searchsorted(y_axis, y_clamped, side="right"))
    x_hi = min(max(1, x_hi), len(x_axis) - 1)
    y_hi = min(max(1, y_hi), len(y_axis) - 1)
    x_lo = x_hi - 1
    y_lo = y_hi - 1

    x0 = float(x_axis[x_lo])
    x1 = float(x_axis[x_hi])
    y0 = float(y_axis[y_lo])
    y1 = float(y_axis[y_hi])
    tx = 0.0 if abs(x1 - x0) < 1e-12 else (x_clamped - x0) / (x1 - x0)
    ty = 0.0 if abs(y1 - y0) < 1e-12 else (y_clamped - y0) / (y1 - y0)

    v00 = float(values[y_lo, x_lo])
    v10 = float(values[y_lo, x_hi])
    v01 = float(values[y_hi, x_lo])
    v11 = float(values[y_hi, x_hi])
    return float(
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )


def find_optimal_grey_point(
    xyz_to_rgb_model: XYZToRGBMLP,
    white_xyz: np.ndarray,
    grey_L: float,
    initial_bounds: tuple[float, float, float, float],
    grid_size: int,
    n_refinements: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    u_lo, u_hi, v_lo, v_hi = initial_bounds
    best_info: dict[str, Any] | None = None
    saved_grid_rows: pd.DataFrame | None = None

    for refinement in range(int(max(1, n_refinements))):
        u_values = np.linspace(u_lo, u_hi, int(grid_size), dtype=float)
        v_values = np.linspace(v_lo, v_hi, int(grid_size), dtype=float)
        du = float(u_values[1] - u_values[0])
        dv = float(v_values[1] - v_values[0])

        valid_mask, xyz_grid, raw_rgb_grid = evaluate_luv_gamut_mask(
            xyz_to_rgb_model,
            white_xyz,
            grey_L,
            u_values,
            v_values,
        )
        distance_map = compute_valid_distance_map(valid_mask, du=du, dv=dv)
        if not np.any(valid_mask):
            raise RuntimeError(
                "No valid in-gamut points were found on the requested grey-L slice."
            )

        best_index = np.unravel_index(np.argmax(distance_map), distance_map.shape)
        best_v_idx, best_u_idx = int(best_index[0]), int(best_index[1])
        best_u = float(u_values[best_u_idx])
        best_v = float(v_values[best_v_idx])
        best_radius = float(distance_map[best_v_idx, best_u_idx])
        best_xyz = xyz_grid[best_v_idx, best_u_idx, :]
        best_rgb_bundle = xyz_to_rgb_model.predict_rgb(best_xyz, warn=False, clip=False)
        best_raw_rgb = np.asarray(best_rgb_bundle.rgb_float, dtype=float).reshape(3)
        best_rgb_int = np.asarray(best_rgb_bundle.rgb_int, dtype=int).reshape(3)

        best_info = {
            "refinement": refinement,
            "grey_L": float(grey_L),
            "grey_u": best_u,
            "grey_v": best_v,
            "circle_radius_uv": best_radius,
            "grid_step_u": du,
            "grid_step_v": dv,
            "u_bounds": [float(u_lo), float(u_hi)],
            "v_bounds": [float(v_lo), float(v_hi)],
            "grey_X": float(best_xyz[0]),
            "grey_Y": float(best_xyz[1]),
            "grey_Z": float(best_xyz[2]),
            "grey_r_float": float(best_raw_rgb[0]),
            "grey_g_float": float(best_raw_rgb[1]),
            "grey_b_float": float(best_raw_rgb[2]),
            "grey_r_int": int(best_rgb_int[0]),
            "grey_g_int": int(best_rgb_int[1]),
            "grey_b_int": int(best_rgb_int[2]),
            "grey_in_gamut": bool((best_raw_rgb >= 0.0).all() and (best_raw_rgb <= 255.0).all()),
            "valid_fraction": float(np.mean(valid_mask)),
        }

        if refinement == int(max(1, n_refinements)) - 1:
            uu, vv = np.meshgrid(u_values, v_values, indexing="xy")
            saved_grid_rows = pd.DataFrame(
                {
                    "u": uu.ravel(),
                    "v": vv.ravel(),
                    "valid": valid_mask.ravel().astype(int),
                    "distance_to_boundary": distance_map.ravel(),
                    "X": xyz_grid[:, :, 0].ravel(),
                    "Y": xyz_grid[:, :, 1].ravel(),
                    "Z": xyz_grid[:, :, 2].ravel(),
                    "pred_r_float": raw_rgb_grid[:, :, 0].ravel(),
                    "pred_g_float": raw_rgb_grid[:, :, 1].ravel(),
                    "pred_b_float": raw_rgb_grid[:, :, 2].ravel(),
                }
            )

        shrink_u = max(best_radius * 2.5, du * 12.0)
        shrink_v = max(best_radius * 2.5, dv * 12.0)
        u_lo = best_u - shrink_u
        u_hi = best_u + shrink_u
        v_lo = best_v - shrink_v
        v_hi = best_v + shrink_v

    if best_info is None or saved_grid_rows is None:
        raise RuntimeError("Grey point optimization failed to produce a valid result.")
    return best_info, saved_grid_rows


def evaluate_fixed_grey_point(
    xyz_to_rgb_model: XYZToRGBMLP,
    white_xyz: np.ndarray,
    grey_xyz: np.ndarray,
    initial_bounds: tuple[float, float, float, float],
    grid_size: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    grey_xyz = np.asarray(grey_xyz, dtype=float).reshape(3)
    if not np.isfinite(grey_xyz).all():
        raise ValueError("set-grey-xyz must contain only finite values.")
    if np.any(grey_xyz < 0.0):
        raise ValueError("set-grey-xyz must be non-negative in XYZ.")

    grey_luv = xyz_to_cie_luv(grey_xyz.reshape(1, 3), white_xyz)[0]
    grey_L = float(grey_luv[0])
    grey_u = float(grey_luv[1])
    grey_v = float(grey_luv[2])

    u_lo, u_hi, v_lo, v_hi = initial_bounds
    u_span = max(float(u_hi - u_lo), 1.0)
    v_span = max(float(v_hi - v_lo), 1.0)
    u_lo = min(float(u_lo), grey_u - 0.08 * u_span)
    u_hi = max(float(u_hi), grey_u + 0.08 * u_span)
    v_lo = min(float(v_lo), grey_v - 0.08 * v_span)
    v_hi = max(float(v_hi), grey_v + 0.08 * v_span)

    u_values = np.linspace(u_lo, u_hi, int(grid_size), dtype=float)
    v_values = np.linspace(v_lo, v_hi, int(grid_size), dtype=float)
    du = float(u_values[1] - u_values[0])
    dv = float(v_values[1] - v_values[0])

    valid_mask, xyz_grid, raw_rgb_grid = evaluate_luv_gamut_mask(
        xyz_to_rgb_model,
        white_xyz,
        grey_L,
        u_values,
        v_values,
    )
    if not np.any(valid_mask):
        raise RuntimeError(
            "No valid in-gamut points were found on the requested grey-L slice."
        )

    distance_map = compute_valid_distance_map(valid_mask, du=du, dv=dv)
    rgb_info = evaluate_xyz_to_rgb_point(xyz_to_rgb_model, grey_xyz)
    circle_radius = 0.0
    if rgb_info["in_gamut"]:
        circle_radius = max(
            0.0,
            interpolate_grid_value(
                u_values,
                v_values,
                distance_map,
                grey_u,
                grey_v,
            ),
        )

    uu, vv = np.meshgrid(u_values, v_values, indexing="xy")
    grid_rows = pd.DataFrame(
        {
            "u": uu.ravel(),
            "v": vv.ravel(),
            "valid": valid_mask.ravel().astype(int),
            "distance_to_boundary": distance_map.ravel(),
            "X": xyz_grid[:, :, 0].ravel(),
            "Y": xyz_grid[:, :, 1].ravel(),
            "Z": xyz_grid[:, :, 2].ravel(),
            "pred_r_float": raw_rgb_grid[:, :, 0].ravel(),
            "pred_g_float": raw_rgb_grid[:, :, 1].ravel(),
            "pred_b_float": raw_rgb_grid[:, :, 2].ravel(),
        }
    )

    grey_info = {
        "selection_mode": "set_grey_xyz",
        "refinement": None,
        "grey_L": grey_L,
        "grey_u": grey_u,
        "grey_v": grey_v,
        "circle_radius_uv": float(circle_radius),
        "grid_step_u": du,
        "grid_step_v": dv,
        "u_bounds": [float(u_lo), float(u_hi)],
        "v_bounds": [float(v_lo), float(v_hi)],
        "grey_X": float(grey_xyz[0]),
        "grey_Y": float(grey_xyz[1]),
        "grey_Z": float(grey_xyz[2]),
        "grey_r_float": float(rgb_info["pred_r_float"]),
        "grey_g_float": float(rgb_info["pred_g_float"]),
        "grey_b_float": float(rgb_info["pred_b_float"]),
        "grey_r_int": int(rgb_info["pred_r_int"]),
        "grey_g_int": int(rgb_info["pred_g_int"]),
        "grey_b_int": int(rgb_info["pred_b_int"]),
        "grey_in_gamut": bool(rgb_info["in_gamut"]),
        "valid_fraction": float(np.mean(valid_mask)),
        "provided_grey_xyz": [float(grey_xyz[0]), float(grey_xyz[1]), float(grey_xyz[2])],
    }
    return grey_info, grid_rows


def build_luminance_offsets(num_additional_levels: int, step_size: float) -> list[dict[str, Any]]:
    num_additional_levels = int(num_additional_levels)
    if num_additional_levels < 0:
        raise ValueError("num_additional_levels must be >= 0.")
    if num_additional_levels % 2 != 0:
        raise ValueError("num_additional_levels must be even so levels are symmetric around grey.")

    half = num_additional_levels // 2
    offsets = []
    for rank, step_index in enumerate(range(-half, half + 1)):
        offsets.append(
            {
                "level_rank": int(rank),
                "level_index": int(step_index + half),
                "offset_steps": int(step_index),
                "offset_L": float(step_index * step_size),
            }
        )
    return offsets


def evaluate_xyz_to_rgb_point(
    xyz_to_rgb_model: XYZToRGBMLP,
    xyz_point: np.ndarray,
) -> dict[str, Any]:
    xyz = np.asarray(xyz_point, dtype=float).reshape(1, 3)
    bundle = xyz_to_rgb_model.predict_rgb(xyz, warn=False, clip=False)
    raw_rgb = np.asarray(bundle.rgb_float, dtype=float).reshape(3)
    rgb_int = np.asarray(bundle.rgb_int, dtype=int).reshape(3)
    valid_xyz = np.isfinite(xyz).all() and np.all(xyz >= 0.0)
    in_gamut = bool(valid_xyz and np.isfinite(raw_rgb).all() and (raw_rgb >= 0.0).all() and (raw_rgb <= 255.0).all())
    return {
        "pred_r_float": float(raw_rgb[0]),
        "pred_g_float": float(raw_rgb[1]),
        "pred_b_float": float(raw_rgb[2]),
        "pred_r_int": int(rgb_int[0]),
        "pred_g_int": int(rgb_int[1]),
        "pred_b_int": int(rgb_int[2]),
        "in_gamut": in_gamut,
        "out_of_gamut_channels": int(np.sum((raw_rgb < 0.0) | (raw_rgb > 255.0) | (~np.isfinite(raw_rgb)))),
    }


def build_virtual_greys(
    grey_info: dict[str, Any],
    white_xyz: np.ndarray,
    xyz_to_rgb_model: XYZToRGBMLP,
    luminance_offsets: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_u = float(grey_info["grey_u"])
    base_v = float(grey_info["grey_v"])
    for level in luminance_offsets:
        level_L = float(grey_info["grey_L"] + level["offset_L"])
        luv = np.array([[level_L, base_u, base_v]], dtype=float)
        xyz = cie_luv_to_xyz(luv, white_xyz)[0]
        rgb_info = evaluate_xyz_to_rgb_point(xyz_to_rgb_model, xyz)
        row = {
            "kind": "background_grey" if level["offset_steps"] == 0 else "virtual_grey",
            "level_rank": int(level["level_rank"]),
            "level_index": int(level["level_index"]),
            "offset_steps": int(level["offset_steps"]),
            "offset_L": float(level["offset_L"]),
            "grey_L": level_L,
            "grey_u": base_u,
            "grey_v": base_v,
            "grey_X": float(xyz[0]),
            "grey_Y": float(xyz[1]),
            "grey_Z": float(xyz[2]),
        }
        row.update(rgb_info)
        rows.append(row)
    return pd.DataFrame(rows)


def build_target_color_tables(
    target_df: pd.DataFrame,
    grey_levels_df: pd.DataFrame,
    white_xyz: np.ndarray,
    xyz_to_rgb_model: XYZToRGBMLP,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    output_id = 1
    for _, grey_row in grey_levels_df.sort_values("level_rank").iterrows():
        for _, target_row in target_df.sort_values("target_order").iterrows():
            target_L = float(grey_row["grey_L"] + target_row["delta_L"])
            target_u = float(grey_row["grey_u"] + target_row["delta_U"])
            target_v = float(grey_row["grey_v"] + target_row["delta_V"])
            luv = np.array([[target_L, target_u, target_v]], dtype=float)
            xyz = cie_luv_to_xyz(luv, white_xyz)[0]
            rgb_info = evaluate_xyz_to_rgb_point(xyz_to_rgb_model, xyz)
            row = {
                "output_id": int(output_id),
                "base_color_id": int(target_row["base_color_id"]),
                "description": str(target_row["description"]),
                "target_order": int(target_row["target_order"]),
                "level_rank": int(grey_row["level_rank"]),
                "level_index": int(grey_row["level_index"]),
                "offset_steps": int(grey_row["offset_steps"]),
                "offset_L": float(grey_row["offset_L"]),
                "grey_L": float(grey_row["grey_L"]),
                "grey_u": float(grey_row["grey_u"]),
                "grey_v": float(grey_row["grey_v"]),
                "delta_L": float(target_row["delta_L"]),
                "delta_U": float(target_row["delta_U"]),
                "delta_V": float(target_row["delta_V"]),
                "target_L": target_L,
                "target_u": target_u,
                "target_v": target_v,
                "target_X": float(xyz[0]),
                "target_Y": float(xyz[1]),
                "target_Z": float(xyz[2]),
            }
            row.update(rgb_info)
            rows.append(row)
            output_id += 1
    return pd.DataFrame(rows)


def build_required_output_tables(
    grey_levels_df: pd.DataFrame,
    color_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    background = grey_levels_df.loc[grey_levels_df["offset_steps"] == 0].iloc[0]
    rgb_rows = [
        {
            "id": 0,
            "r": int(background["pred_r_int"]),
            "g": int(background["pred_g_int"]),
            "b": int(background["pred_b_int"]),
        }
    ]
    xyz_rows = [
        {
            "id": 0,
            "x": float(background["grey_X"]),
            "y": float(background["grey_Y"]),
            "z": float(background["grey_Z"]),
        }
    ]

    ordered = color_df.sort_values(["level_rank", "target_order", "output_id"]).reset_index(drop=True)
    for _, row in ordered.iterrows():
        rgb_rows.append(
            {
                "id": int(row["output_id"]),
                "r": int(row["pred_r_int"]),
                "g": int(row["pred_g_int"]),
                "b": int(row["pred_b_int"]),
            }
        )
        xyz_rows.append(
            {
                "id": int(row["output_id"]),
                "x": float(row["target_X"]),
                "y": float(row["target_Y"]),
                "z": float(row["target_Z"]),
            }
        )

    return pd.DataFrame(rgb_rows), pd.DataFrame(xyz_rows)


def plot_gamut_luv_cloud(gamut_df: pd.DataFrame, out_path: Path, *, show: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    sc = ax.scatter(
        gamut_df["u"],
        gamut_df["v"],
        c=gamut_df["L"],
        s=12,
        cmap="viridis",
        alpha=0.65,
        linewidths=0.0,
    )
    ax.set_title("Sampled monitor gamut in CIE-LUV")
    ax.set_xlabel("u*")
    ax.set_ylabel("v*")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("L*")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_grey_slice(
    grey_grid_df: pd.DataFrame,
    slice_points_df: pd.DataFrame,
    grey_info: dict[str, Any],
    out_path: Path,
    *,
    show: bool = False,
) -> None:
    pivot_valid = grey_grid_df.pivot(index="v", columns="u", values="valid").sort_index()
    u_values = pivot_valid.columns.to_numpy(dtype=float)
    v_values = pivot_valid.index.to_numpy(dtype=float)
    valid_mask = pivot_valid.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    ax.imshow(
        valid_mask,
        origin="lower",
        extent=[u_values.min(), u_values.max(), v_values.min(), v_values.max()],
        cmap="Greys",
        alpha=0.72,
        aspect="auto",
    )
    ax.scatter(
        slice_points_df["u"],
        slice_points_df["v"],
        s=18,
        c=slice_points_df["L"],
        cmap="viridis",
        alpha=0.55,
        label="sampled RGB gamut points near grey L",
    )

    circle = plt.Circle(
        (grey_info["grey_u"], grey_info["grey_v"]),
        grey_info["circle_radius_uv"],
        color="tab:red",
        fill=False,
        linewidth=2.0,
        label=(
            "largest inscribed circle"
            if grey_info.get("selection_mode", "optimized") == "optimized"
            else "available radius around selected grey"
        ),
    )
    ax.add_patch(circle)
    ax.scatter(
        [grey_info["grey_u"]],
        [grey_info["grey_v"]],
        color="tab:red",
        s=50,
        marker="x",
        linewidths=2.0,
        label=(
            "optimal grey point"
            if grey_info.get("selection_mode", "optimized") == "optimized"
            else "selected grey point"
        ),
    )
    title_prefix = (
        "Grey-L slice optimization"
        if grey_info.get("selection_mode", "optimized") == "optimized"
        else "Grey-L slice with provided background grey"
    )
    ax.set_title(f"{title_prefix} (L*={grey_info['grey_L']:.2f}, radius={grey_info['circle_radius_uv']:.3f})")
    ax.set_xlabel("u*")
    ax.set_ylabel("v*")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_gamut_slices_across_L(
    xyz_to_rgb_model: XYZToRGBMLP,
    white_xyz: np.ndarray,
    gamut_df: pd.DataFrame,
    base_bounds: tuple[float, float, float, float],
    grey_info: dict[str, Any],
    out_path: Path,
    *,
    show: bool = False,
) -> None:
    levels = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    u_lo, u_hi, v_lo, v_hi = base_bounds
    u_values = np.linspace(u_lo, u_hi, 181, dtype=float)
    v_values = np.linspace(v_lo, v_hi, 181, dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.8), squeeze=False)
    for ax, level_L in zip(axes.ravel(), levels, strict=True):
        valid_mask, _, _ = evaluate_luv_gamut_mask(
            xyz_to_rgb_model,
            white_xyz,
            level_L=level_L,
            u_values=u_values,
            v_values=v_values,
        )
        ax.imshow(
            valid_mask.astype(float),
            origin="lower",
            extent=[u_values.min(), u_values.max(), v_values.min(), v_values.max()],
            cmap="Greys",
            alpha=0.72,
            aspect="auto",
        )

        tol = 1.5 if level_L not in (0.0, 100.0) else 3.0
        subset = gamut_df[np.abs(gamut_df["L"] - level_L) <= tol].copy()
        if not subset.empty:
            ax.scatter(
                subset["u"],
                subset["v"],
                s=10,
                c=subset["L"],
                cmap="viridis",
                alpha=0.45,
                linewidths=0.0,
            )

        if abs(level_L - float(grey_info["grey_L"])) <= 10.0:
            ax.scatter(
                [grey_info["grey_u"]],
                [grey_info["grey_v"]],
                color="tab:red",
                s=38,
                marker="x",
                linewidths=1.8,
            )

        ax.set_title(f"L* = {level_L:.0f}")
        ax.set_xlabel("u*")
        ax.set_ylabel("v*")
        ax.grid(alpha=0.2)

    fig.suptitle("Monitor gamut slices in CIE-LUV at every 20 L*", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_out_of_gamut_summary(
    color_df: pd.DataFrame,
    out_path: Path,
    *,
    show: bool = False,
) -> None:
    ordered_levels = color_df[["level_rank", "grey_L"]].drop_duplicates().sort_values("level_rank")
    n_levels = len(ordered_levels)
    ncols = min(4, max(1, n_levels))
    nrows = int(math.ceil(n_levels / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 4.0 * nrows),
        squeeze=False,
    )

    for ax in axes.ravel():
        ax.set_visible(False)

    for idx, (_, level_row) in enumerate(ordered_levels.iterrows()):
        ax = axes[idx // ncols, idx % ncols]
        ax.set_visible(True)
        subset = color_df[color_df["level_rank"] == int(level_row["level_rank"])].copy()
        in_gamut = subset["in_gamut"].to_numpy(dtype=bool)
        ax.scatter(
            subset.loc[in_gamut, "delta_U"],
            subset.loc[in_gamut, "delta_V"],
            color="tab:green",
            s=42,
            alpha=0.9,
            label="in gamut",
        )
        ax.scatter(
            subset.loc[~in_gamut, "delta_U"],
            subset.loc[~in_gamut, "delta_V"],
            color="tab:red",
            s=58,
            marker="x",
            linewidths=1.8,
            label="out of gamut",
        )
        ax.scatter([0.0], [0.0], color="black", s=24, marker="o")
        ax.set_title(
            f"Level {int(level_row['level_rank'])} | grey L*={float(level_row['grey_L']):.1f} | OOG={(~in_gamut).sum()}"
        )
        ax.set_xlabel("delta_U from level grey")
        ax.set_ylabel("delta_V from level grey")
        ax.grid(alpha=0.25)
        if np.any(~in_gamut):
            for _, bad_row in subset.loc[~in_gamut].iterrows():
                ax.annotate(
                    str(int(bad_row["base_color_id"])),
                    (float(bad_row["delta_U"]), float(bad_row["delta_V"])),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                )
        if idx == 0:
            ax.legend(loc="best", fontsize=8)

    fig.suptitle("Final target colors by luminance level: out-of-gamut summary", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_color_wheels_by_luminance(
    color_df: pd.DataFrame,
    grey_levels_df: pd.DataFrame,
    out_dir: Path,
    *,
    filename_suffix: str = "",
    show: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    real_grey = grey_levels_df.loc[grey_levels_df["offset_steps"] == 0].iloc[0]
    background_rgb = np.array(
        [
            float(real_grey["pred_r_int"]) / 255.0,
            float(real_grey["pred_g_int"]) / 255.0,
            float(real_grey["pred_b_int"]) / 255.0,
        ],
        dtype=float,
    )

    ordered_levels = color_df[["level_rank", "grey_L"]].drop_duplicates().sort_values("level_rank")
    if ordered_levels.empty:
        return

    radius = 1.0
    for _, level_row in ordered_levels.iterrows():
        subset = (
            color_df[color_df["level_rank"] == int(level_row["level_rank"])]
            .sort_values(["target_order", "output_id"])
            .reset_index(drop=True)
        )
        if subset.empty:
            continue

        n_colors = len(subset)
        theta = (np.pi / 2.0) - (2.0 * np.pi * np.arange(n_colors, dtype=float) / max(n_colors, 1))
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        chord = 2.0 * radius * math.sin(math.pi / max(n_colors, 2))
        marker_radius = min(0.11, max(0.045, 0.36 * chord))

        fig, ax = plt.subplots(figsize=(6.8, 6.8))
        fig.patch.set_facecolor(background_rgb)
        ax.set_facecolor(background_rgb)

        outer_ring = plt.Circle(
            (0.0, 0.0),
            radius + marker_radius * 0.95,
            edgecolor=(0.15, 0.15, 0.15),
            facecolor="none",
            linewidth=0.8,
            alpha=0.65,
        )
        ax.add_patch(outer_ring)

        for idx, (_, row) in enumerate(subset.iterrows()):
            face_rgb = (
                float(row["pred_r_int"]) / 255.0,
                float(row["pred_g_int"]) / 255.0,
                float(row["pred_b_int"]) / 255.0,
            )
            patch = plt.Circle(
                (float(x[idx]), float(y[idx])),
                marker_radius,
                facecolor=face_rgb,
                edgecolor="black",
                linewidth=0.6,
            )
            ax.add_patch(patch)

        ax.set_xlim(-(radius + 0.28), radius + 0.28)
        ax.set_ylim(-(radius + 0.28), radius + 0.28)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title(
            f"Final target wheel | level {int(level_row['level_rank'])} | L*={float(level_row['grey_L']):.1f}",
            color="black",
            pad=12,
        )
        fig.tight_layout()
        out_path = out_dir / (
            f"level_{int(level_row['level_rank']):02d}_L_{float(level_row['grey_L']):05.1f}{filename_suffix}.png"
        )
        fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        if show:
            plt.show()
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Map grey-relative CIE-LUV target definitions to the new screen."
    )
    parser.add_argument(
        "--forward-model-path",
        default=str(DEFAULT_FORWARD_MODEL),
        help=f"Path to the saved XYZ -> RGB MLP. Default: {DEFAULT_FORWARD_MODEL}",
    )
    parser.add_argument(
        "--inverse-model-path",
        default=str(DEFAULT_INVERSE_MODEL),
        help=f"Path to the saved RGB -> XYZ MLP. Default: {DEFAULT_INVERSE_MODEL}",
    )
    parser.add_argument(
        "--measured-xyz-path",
        default=str(DEFAULT_MEASURED_XYZ),
        help=f"Measured XYZ CSV used to derive the white reference. Default: {DEFAULT_MEASURED_XYZ}",
    )
    parser.add_argument(
        "--target-displacements-path",
        default=str(DEFAULT_TARGET_DISPLACEMENTS),
        help=(
            "CSV with at least id,delta_L,delta_U,delta_V describing grey-relative targets. "
            f"Default: {DEFAULT_TARGET_DISPLACEMENTS}"
        ),
    )
    parser.add_argument(
        "--white-id",
        type=int,
        default=1,
        help="White-point ID in the measured XYZ table. Default: 1",
    )
    parser.add_argument(
        "--grey-l",
        type=float,
        default=None,
        help="Absolute L* for the real grey point. If omitted, grey-l-fraction is used.",
    )
    parser.add_argument(
        "--grey-l-fraction",
        type=float,
        default=0.70,
        help="Grey-point L* as a fraction of white L*. Default: 0.70",
    )
    parser.add_argument(
        "--set-grey-xyz",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Override the optimized background grey with a provided XYZ coordinate. "
            "When set, grey L*, u*, and v* are taken from this XYZ in the new screen's white-referenced CIE-LUV space."
        ),
    )
    parser.add_argument(
        "--num-additional-lum-levels",
        type=int,
        default=10,
        help="Even number of total additional luminance levels around grey. Default: 10",
    )
    parser.add_argument(
        "--lum-step-size",
        type=float,
        default=2.0,
        help="L* step size between adjacent luminance levels. Default: 2.0",
    )
    parser.add_argument(
        "--gamut-rgb-grid-size",
        type=int,
        default=21,
        help="Grid size per RGB axis when sampling the full monitor gamut. Default: 21",
    )
    parser.add_argument(
        "--grey-slice-grid-size",
        type=int,
        default=201,
        help="Grid size per axis for grey-point search in the fixed-L u/v slice. Default: 201",
    )
    parser.add_argument(
        "--grey-search-refinements",
        type=int,
        default=3,
        help="Number of multi-resolution grey-point search refinements. Default: 3",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Output directory root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively after saving them.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    data_dir = output_root / "data"
    plots_dir = output_root / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_name_suffix = "_set_grey" if args.set_grey_xyz is not None else ""

    forward_model = XYZToRGBMLP.load_state(Path(args.forward_model_path).resolve())
    inverse_model = RGBToXYZMLP.load_state(Path(args.inverse_model_path).resolve())
    measured_xyz_df = load_measured_xyz_table(Path(args.measured_xyz_path).resolve())
    target_df = load_target_displacements(Path(args.target_displacements_path).resolve())

    white_rows = measured_xyz_df[measured_xyz_df["id"] == int(args.white_id)].copy()
    if white_rows.empty:
        raise ValueError(f"No rows with white_id={args.white_id} found in measured XYZ table.")
    white_xyz = white_rows[["X", "Y", "Z"]].mean(axis=0).to_numpy(dtype=float)
    white_luv = xyz_to_cie_luv(white_xyz.reshape(1, 3), white_xyz)[0]
    white_L = float(white_luv[0])

    if args.grey_l is not None:
        grey_L = float(args.grey_l)
    else:
        grey_L = float(args.grey_l_fraction) * white_L

    gamut_df = compute_gamut_samples(
        inverse_model,
        white_xyz=white_xyz,
        rgb_grid_size=int(args.gamut_rgb_grid_size),
    )
    gamut_csv_path = apply_name_suffix(data_dir / "sampled_monitor_gamut_rgb_xyz_luv.csv", output_name_suffix)
    gamut_df.to_csv(gamut_csv_path, index=False)

    grey_selection_mode = "optimized"
    if args.set_grey_xyz is not None:
        grey_selection_mode = "set_grey_xyz"
        provided_grey_xyz = np.asarray(args.set_grey_xyz, dtype=float).reshape(3)
        provided_grey_luv = xyz_to_cie_luv(provided_grey_xyz.reshape(1, 3), white_xyz)[0]
        grey_L = float(provided_grey_luv[0])

    search_bounds, slice_points_df, used_slice_tol = select_l_slice_bounds(gamut_df, grey_L=grey_L)
    slice_points_csv_path = apply_name_suffix(data_dir / "sampled_gamut_points_near_grey_slice.csv", output_name_suffix)
    slice_points_df.to_csv(slice_points_csv_path, index=False)

    if args.set_grey_xyz is not None:
        grey_info, grey_grid_df = evaluate_fixed_grey_point(
            forward_model,
            white_xyz=white_xyz,
            grey_xyz=np.asarray(args.set_grey_xyz, dtype=float),
            initial_bounds=search_bounds,
            grid_size=int(args.grey_slice_grid_size),
        )
    else:
        grey_info, grey_grid_df = find_optimal_grey_point(
            forward_model,
            white_xyz=white_xyz,
            grey_L=grey_L,
            initial_bounds=search_bounds,
            grid_size=int(args.grey_slice_grid_size),
            n_refinements=int(args.grey_search_refinements),
        )
        grey_info["selection_mode"] = "optimized"
    grey_info["white_X"] = float(white_xyz[0])
    grey_info["white_Y"] = float(white_xyz[1])
    grey_info["white_Z"] = float(white_xyz[2])
    grey_info["white_L"] = white_L
    grey_info["grey_slice_tolerance_used"] = float(used_slice_tol)
    grey_info["output_name_suffix"] = output_name_suffix
    grey_info["grey_selection_mode"] = grey_selection_mode
    optimal_grey_path = apply_name_suffix(data_dir / "optimal_grey_point.json", output_name_suffix)
    with optimal_grey_path.open("w", encoding="utf-8") as handle:
        json.dump(grey_info, handle, indent=2, sort_keys=True)
    grey_grid_csv_path = apply_name_suffix(data_dir / "grey_slice_search_grid.csv", output_name_suffix)
    grey_grid_df.to_csv(grey_grid_csv_path, index=False)

    luminance_offsets = build_luminance_offsets(
        num_additional_levels=int(args.num_additional_lum_levels),
        step_size=float(args.lum_step_size),
    )
    grey_levels_df = build_virtual_greys(
        grey_info=grey_info,
        white_xyz=white_xyz,
        xyz_to_rgb_model=forward_model,
        luminance_offsets=luminance_offsets,
    )
    grey_levels_csv_path = apply_name_suffix(data_dir / "virtual_grey_levels.csv", output_name_suffix)
    grey_levels_df.to_csv(grey_levels_csv_path, index=False)

    color_df = build_target_color_tables(
        target_df=target_df,
        grey_levels_df=grey_levels_df,
        white_xyz=white_xyz,
        xyz_to_rgb_model=forward_model,
    )
    detailed_colors_csv_path = apply_name_suffix(data_dir / "new_screen_target_colors_detailed.csv", output_name_suffix)
    color_df.to_csv(detailed_colors_csv_path, index=False)

    rgb_output_df, xyz_output_df = build_required_output_tables(grey_levels_df, color_df)
    rgb_output_path = apply_name_suffix(data_dir / "new_screen_colors_rgb.tsv", output_name_suffix)
    xyz_output_path = apply_name_suffix(data_dir / "new_screen_colors_xyz.tsv", output_name_suffix)
    rgb_output_df.to_csv(rgb_output_path, sep="\t", index=False)
    xyz_output_df.to_csv(xyz_output_path, sep="\t", index=False)

    out_of_gamut_summary_df = (
        color_df.groupby(["level_rank", "grey_L"], as_index=False)
        .agg(
            n_colors=("output_id", "size"),
            n_out_of_gamut=("in_gamut", lambda x: int((~x).sum())),
            max_out_of_gamut_channels=("out_of_gamut_channels", "max"),
        )
        .sort_values("level_rank")
        .reset_index(drop=True)
    )
    out_of_gamut_summary_path = apply_name_suffix(data_dir / "out_of_gamut_summary.csv", output_name_suffix)
    out_of_gamut_summary_df.to_csv(out_of_gamut_summary_path, index=False)

    summary = {
        "white_reference_xyz": white_xyz.tolist(),
        "white_reference_L": white_L,
        "grey_info": grey_info,
        "grey_selection_mode": grey_selection_mode,
        "set_grey_xyz": list(map(float, args.set_grey_xyz)) if args.set_grey_xyz is not None else None,
        "output_name_suffix": output_name_suffix,
        "n_base_colors": int(len(target_df)),
        "n_luminance_levels": int(len(grey_levels_df)),
        "n_selectable_colors": int(len(color_df)),
        "n_out_of_gamut_colors": int((~color_df["in_gamut"]).sum()),
        "max_out_of_gamut_channels": int(color_df["out_of_gamut_channels"].max() if len(color_df) else 0),
        "rgb_output_path": str(rgb_output_path.resolve()),
        "xyz_output_path": str(xyz_output_path.resolve()),
    }
    run_summary_path = apply_name_suffix(data_dir / "run_summary.json", output_name_suffix)
    with run_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    plot_gamut_luv_cloud(
        gamut_df,
        apply_name_suffix(plots_dir / "01_sampled_gamut_luv_cloud.png", output_name_suffix),
        show=bool(args.show_plots),
    )
    plot_grey_slice(
        grey_grid_df,
        slice_points_df,
        grey_info,
        apply_name_suffix(plots_dir / "02_grey_slice_and_inscribed_circle.png", output_name_suffix),
        show=bool(args.show_plots),
    )
    plot_gamut_slices_across_L(
        forward_model,
        white_xyz=white_xyz,
        gamut_df=gamut_df,
        base_bounds=search_bounds,
        grey_info=grey_info,
        out_path=apply_name_suffix(plots_dir / "03_gamut_slices_every_20L.png", output_name_suffix),
        show=bool(args.show_plots),
    )
    plot_out_of_gamut_summary(
        color_df,
        apply_name_suffix(plots_dir / "04_out_of_gamut_summary.png", output_name_suffix),
        show=bool(args.show_plots),
    )
    plot_color_wheels_by_luminance(
        color_df,
        grey_levels_df,
        apply_name_suffix(plots_dir / "05_color_wheels_by_luminance", output_name_suffix),
        filename_suffix=output_name_suffix,
        show=bool(args.show_plots),
    )

    if int(args.num_additional_lum_levels) % 2 != 0:
        raise RuntimeError("num_additional_lum_levels validation should have failed earlier.")

    print("Saved RGB output to:", rgb_output_path)
    print("Saved XYZ output to:", xyz_output_path)
    print(
        "Selected grey point:",
        f"L*={grey_info['grey_L']:.3f}",
        f"u*={grey_info['grey_u']:.3f}",
        f"v*={grey_info['grey_v']:.3f}",
        f"radius={grey_info['circle_radius_uv']:.3f}",
    )
    print(
        "Background grey RGB:",
        f"({grey_info['grey_r_int']}, {grey_info['grey_g_int']}, {grey_info['grey_b_int']})",
    )
    print(
        "Selectable colors:",
        len(color_df),
        "| Out of gamut:",
        int((~color_df["in_gamut"]).sum()),
    )

    if (~color_df["in_gamut"]).any():
        warnings.warn(
            f"{int((~color_df['in_gamut']).sum())} final target colors are out of gamut and were clipped in RGB output.",
            RuntimeWarning,
            stacklevel=2,
        )


if __name__ == "__main__":
    main()
