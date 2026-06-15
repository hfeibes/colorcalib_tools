#!/usr/bin/env python3
"""
Construct a historical grey-relative CIE-LUV target from measured XYZ files.

This script is self-contained inside `calibration/` and is intended to be the
first stage of a more formal calibration pipeline.

Inputs
------
- One or more input tables, each either:
  - raw spectra with columns: rep,id,r,g,b,nm,power
  - measured XYZ with columns: rep,id,r,g,b,X,Y,Z

Raw spectra are converted to XYZ internally using the requested color matching
functions. The default is true XYZ 1931 (`ciexyz31.txt`).

Per source file and repetition:
- white is identified by `--white-id`
- black is identified by `--black-id` (recorded, not used in the fit)
- grey is identified by `--grey-id`

Each valid color sample is converted to white-referenced CIE L*u*v*, then
converted to a grey-relative chromatic vector used for the ellipse fit:

    U_fit = 13 * L_norm * (u'_color - u'_grey)
    V_fit = 13 * L_norm * (v'_color - v'_grey)

where `L_norm` defaults to 100. This keeps the hue geometry comparable across
mixed luminance samples while still saving the raw white-referenced CIE-LUV
values and grey-relative deltas for inspection.

Final target points are solved so adjacent colors have equal Euclidean
distance in the 2D fit plane. Arc length between neighboring colors on the
ellipse is recorded as metadata and is allowed to vary.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize, minimize_scalar


DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parent
    / "outputs"
    / "01_establish_historical_cieluv_target"
)
DEFAULT_CMF_1931 = (
    Path(__file__).resolve().parents[1] / "color_matching_functions" / "ciexyz31.txt"
)
DEFAULT_CMF_JUDD = (
    Path(__file__).resolve().parents[1] / "color_matching_functions" / "ciexyzj.txt"
)

SPD_COLUMNS = {"rep", "id", "r", "g", "b", "nm", "power"}
XYZ_COLUMNS = {"rep", "id", "r", "g", "b", "X", "Y", "Z"}
REQUIRED_COLUMNS = XYZ_COLUMNS


@dataclass(frozen=True)
class EllipseFit:
    center_u: float
    center_v: float
    axis_a: float
    axis_b: float
    axis_ratio: float
    angle_rad: float
    rmse: float
    objective: float
    iterations: int
    converged: bool


def parse_id_spec(spec: str | None) -> list[int]:
    if spec is None:
        return []
    text = str(spec).strip()
    if not text:
        return []

    values: list[int] = []
    for chunk in text.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        if "-" in piece:
            start_text, end_text = piece.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            step = 1 if end >= start else -1
            values.extend(list(range(start, end + step, step)))
        else:
            values.append(int(piece))
    return values


def parse_handle_groups(spec: str | None) -> list[list[int]]:
    if spec is None:
        return []
    text = str(spec).strip()
    if not text:
        return []
    groups: list[list[int]] = []
    for group_text in text.split(";"):
        ids = parse_id_spec(group_text)
        if not ids:
            continue
        groups.append(ids)
    return groups


def parse_labels(spec: str | None) -> list[str]:
    if spec is None:
        return []
    text = str(spec).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def read_table_auto(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def normalize_cmf_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "").replace("_", "")
    aliases_1931 = {"1931", "cie1931", "ciexyz1931", "xyz1931", "cie31", "31"}
    aliases_judd = {"judd", "ciejudd", "ciejudd1951", "judd1951", "ciexyzj"}
    if key in aliases_1931:
        return "cie1931"
    if key in aliases_judd:
        return "cie_judd"
    raise ValueError(
        f"Unsupported CMF '{name}'. Supported values include: 1931, cie1931, judd."
    )


def load_cmf(cmf_name: str, cmf_file: str | Path | None = None) -> tuple[str, Path, np.ndarray, np.ndarray]:
    key = normalize_cmf_name(cmf_name)
    if cmf_file is not None:
        path = Path(cmf_file)
    elif key == "cie1931":
        path = DEFAULT_CMF_1931
    else:
        path = DEFAULT_CMF_JUDD

    cmf_df = pd.read_csv(
        path,
        header=None,
        names=["wavelength", "x_bar", "y_bar", "z_bar"],
    )
    wavelengths = cmf_df["wavelength"].to_numpy(dtype=float)
    values = cmf_df[["x_bar", "y_bar", "z_bar"]].to_numpy(dtype=float)
    return key, path, wavelengths, values


def interp_to_1nm_domain(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1:
        raise ValueError("Wavelength array must be 1D.")
    if len(x) < 2:
        raise ValueError("Need at least two wavelength samples for interpolation.")

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if y.ndim == 1:
        y = y[:, None]

    if np.max(np.diff(x)) <= 1.0:
        return x, y

    xi = np.arange(int(np.ceil(x[0])), int(np.floor(x[-1])) + 1, 1, dtype=float)
    spline = CubicSpline(x, y, axis=0)
    yi = spline(xi)
    return xi, yi


def common_domain(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1i, y1i = interp_to_1nm_domain(x1, y1)
    x2i, y2i = interp_to_1nm_domain(x2, y2)

    start = max(float(x1i[0]), float(x2i[0]))
    end = min(float(x1i[-1]), float(x2i[-1]))
    if end < start:
        raise ValueError("No overlapping wavelength domain between CMF and spectrum.")

    xc = np.arange(int(np.ceil(start)), int(np.floor(end)) + 1, 1, dtype=float)
    if xc.size == 0:
        raise ValueError("Overlapping wavelength domain is empty.")

    mask1 = np.isin(x1i, xc)
    mask2 = np.isin(x2i, xc)
    y1c = y1i[mask1]
    y2c = y2i[mask2]
    if y1c.shape[0] != xc.shape[0] or y2c.shape[0] != xc.shape[0]:
        raise ValueError("Failed to align CMF and spectra on a shared 1 nm grid.")
    return xc, y1c, y2c[:, 0]


def convert_spd_df_to_xyz(
    spd_df: pd.DataFrame,
    cmf_wavelengths: np.ndarray,
    cmf_values: np.ndarray,
    source_file: str,
) -> pd.DataFrame:
    missing = SPD_COLUMNS - set(spd_df.columns)
    if missing:
        raise ValueError(f"{source_file} is missing required SPD columns: {sorted(missing)}")

    work = spd_df.copy()
    for column in ["rep", "id", "r", "g", "b", "nm", "power"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["rep", "id", "r", "g", "b", "nm", "power"]).copy()
    work["rep"] = work["rep"].astype(int)
    work["id"] = work["id"].astype(int)

    rows: list[list[float | int | str]] = []
    for (rep, color_id), group in work.groupby(["rep", "id"], sort=False):
        group = group.sort_values("nm")
        r = float(group["r"].iloc[0])
        g = float(group["g"].iloc[0])
        b = float(group["b"].iloc[0])
        if r < 0.0 or g < 0.0 or b < 0.0:
            rows.append([rep, color_id, r, g, b, -1.0, -1.0, -1.0, source_file])
            continue

        wavelengths = group["nm"].to_numpy(dtype=float)
        power = group["power"].to_numpy(dtype=float)
        _, cmf_common, spd_common = common_domain(
            cmf_wavelengths,
            cmf_values,
            wavelengths,
            power,
        )
        xyz = spd_common @ cmf_common
        rows.append(
            [
                int(rep),
                int(color_id),
                r,
                g,
                b,
                float(xyz[0]),
                float(xyz[1]),
                float(xyz[2]),
                source_file,
            ]
        )

    return pd.DataFrame(
        rows,
        columns=["rep", "id", "r", "g", "b", "X", "Y", "Z", "source_file"],
    )


def normalize_and_validate(df: pd.DataFrame, source_file: str, tablet_index: int) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{source_file} is missing required columns: {sorted(missing)}")

    out = df.copy()
    for column in ["rep", "id", "r", "g", "b", "X", "Y", "Z"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["rep", "id", "X", "Y", "Z"]).copy()

    out["rep"] = out["rep"].astype(int)
    out["id"] = out["id"].astype(int)
    out["tablet_index"] = int(tablet_index)
    out["source_file"] = source_file
    out["valid_xyz"] = np.isfinite(out[["X", "Y", "Z"]]).all(axis=1) & (out[["X", "Y", "Z"]] >= 0.0).all(axis=1)
    out["valid_rgb"] = np.isfinite(out[["r", "g", "b"]]).all(axis=1) & (out[["r", "g", "b"]] >= 0.0).all(axis=1)
    return out


def xyz_to_upvp(xyz_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz_values, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz_values must be an Nx3 array.")

    den = xyz[:, 0] + 15.0 * xyz[:, 1] + 3.0 * xyz[:, 2]
    up = np.where(den > 1e-12, 4.0 * xyz[:, 0] / den, np.nan)
    vp = np.where(den > 1e-12, 9.0 * xyz[:, 1] / den, np.nan)
    return up, vp


def xyz_to_cie_luv(xyz_values: np.ndarray, reference_xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz_values, dtype=float)
    reference = np.asarray(reference_xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz_values must be an Nx3 array.")
    if reference.shape != (3,):
        raise ValueError("reference_xyz must be length 3.")

    Xn, Yn, Zn = reference
    if Yn <= 0:
        raise ValueError("Reference Y must be > 0.")

    den_ref = Xn + 15.0 * Yn + 3.0 * Zn
    if den_ref <= 0:
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
    L = np.where(yr > epsilon, 116.0 * np.cbrt(yr) - 16.0, kappa * yr)
    u = 13.0 * L * (u_prime - u_n)
    v = 13.0 * L * (v_prime - v_n)
    return np.column_stack([L, u, v])


def ellipse_points(params: EllipseFit | dict[str, float], t_values: np.ndarray) -> np.ndarray:
    t = np.asarray(t_values, dtype=float)
    if isinstance(params, EllipseFit):
        a = params.axis_a
        b = params.axis_b
        phi = params.angle_rad
        cu = params.center_u
        cv = params.center_v
    else:
        a = float(params["axis_a"])
        b = float(params["axis_b"])
        phi = float(params["angle_rad"])
        cu = float(params.get("center_u", 0.0))
        cv = float(params.get("center_v", 0.0))

    c = math.cos(phi)
    s = math.sin(phi)
    ct = np.cos(t)
    st = np.sin(t)

    u = cu + a * ct * c - b * st * s
    v = cv + a * ct * s + b * st * c
    return np.column_stack([u, v])


def fit_origin_centered_ellipse(
    uv_values: np.ndarray,
    circularity_weight: float,
    max_iter: int,
    weights: np.ndarray | None = None,
) -> tuple[EllipseFit, pd.DataFrame]:
    pts = np.asarray(uv_values, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("uv_values must be an Nx2 array.")
    if pts.shape[0] < 3:
        raise ValueError("Need at least 3 valid vectors to fit an ellipse.")
    if circularity_weight < 0:
        raise ValueError("circularity_weight must be >= 0.")
    if weights is None:
        w = np.ones(pts.shape[0], dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != pts.shape[0]:
            raise ValueError("weights must have the same length as uv_values.")
        if np.any(w <= 0):
            raise ValueError("weights must be strictly positive.")
    w = w / np.sum(w)

    cov = np.cov(pts.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    phi0 = float(np.arctan2(evecs[1, 0], evecs[0, 0]))
    std_u = max(float(np.std(pts[:, 0])), 1e-3)
    std_v = max(float(np.std(pts[:, 1])), 1e-3)
    a0 = max(math.sqrt(max(float(evals[0]), 1e-6)) * math.sqrt(2.0), std_u)
    b0 = max(math.sqrt(max(float(evals[1]), 1e-6)) * math.sqrt(2.0), std_v)
    x0 = np.array([math.log(a0), math.log(b0), phi0], dtype=float)
    eps = 1e-12

    def objective(x: np.ndarray) -> float:
        log_a, log_b, phi = x
        a = math.exp(float(log_a))
        b = math.exp(float(log_b))
        c = math.cos(phi)
        s = math.sin(phi)
        x_rot = c * pts[:, 0] + s * pts[:, 1]
        y_rot = -s * pts[:, 0] + c * pts[:, 1]
        rho = np.sqrt((x_rot / a) ** 2 + (y_rot / b) ** 2 + eps)
        mse = np.sum(w * (rho - 1.0) ** 2)
        reg = circularity_weight * (log_a - log_b) ** 2
        return float(mse + reg)

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        options={"maxiter": int(max_iter)},
    )

    log_a, log_b, phi = result.x
    a = float(math.exp(float(log_a)))
    b = float(math.exp(float(log_b)))
    if b > a:
        a, b = b, a
        phi = float(phi + math.pi / 2.0)

    c = math.cos(phi)
    s = math.sin(phi)
    x_rot = c * pts[:, 0] + s * pts[:, 1]
    y_rot = -s * pts[:, 0] + c * pts[:, 1]
    rho = np.sqrt((x_rot / a) ** 2 + (y_rot / b) ** 2 + eps)
    residuals = rho - 1.0

    fit = EllipseFit(
        center_u=0.0,
        center_v=0.0,
        axis_a=float(a),
        axis_b=float(b),
        axis_ratio=float(a / b) if b > 0 else float("inf"),
        angle_rad=float((phi + math.pi) % (2.0 * math.pi) - math.pi),
        rmse=float(np.sqrt(np.mean(residuals**2))),
        objective=float(objective(np.array([math.log(a), math.log(b), phi], dtype=float))),
        iterations=int(getattr(result, "nit", -1)),
        converged=bool(result.success),
    )

    table = pd.DataFrame(
        {
            "fit_u": pts[:, 0],
            "fit_v": pts[:, 1],
            "weight": w,
            "rho": rho,
            "ellipse_residual": residuals,
        }
    )
    return fit, table


def ellipse_residuals_for_points(points: np.ndarray, ellipse: EllipseFit) -> pd.DataFrame:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an Nx2 array.")

    eps = 1e-12
    c = math.cos(ellipse.angle_rad)
    s = math.sin(ellipse.angle_rad)
    x_rot = c * pts[:, 0] + s * pts[:, 1]
    y_rot = -s * pts[:, 0] + c * pts[:, 1]
    rho = np.sqrt((x_rot / ellipse.axis_a) ** 2 + (y_rot / ellipse.axis_b) ** 2 + eps)
    residuals = rho - 1.0
    return pd.DataFrame(
        {
            "rho": rho,
            "ellipse_residual": residuals,
        }
    )


def project_points_to_ellipse(points: np.ndarray, ellipse: EllipseFit, grid_size: int = 2048) -> pd.DataFrame:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an Nx2 array.")
    if grid_size < 64:
        raise ValueError("grid_size must be >= 64.")

    ts = np.linspace(0.0, 2.0 * math.pi, int(grid_size), endpoint=False)
    curve = ellipse_points(ellipse, ts)
    step = 2.0 * math.pi / float(grid_size)
    rows: list[dict[str, float]] = []

    for point in pts:
        d2 = np.sum((curve - point[None, :]) ** 2, axis=1)
        i0 = int(np.argmin(d2))
        t0 = float(ts[i0])

        def distance_squared(t_value: float) -> float:
            q = ellipse_points(ellipse, np.array([t_value], dtype=float))[0]
            return float(np.sum((q - point) ** 2))

        res = minimize_scalar(
            distance_squared,
            bounds=(t0 - 2.0 * step, t0 + 2.0 * step),
            method="bounded",
        )
        t_best = float(res.x % (2.0 * math.pi))
        q_best = ellipse_points(ellipse, np.array([t_best], dtype=float))[0]
        rows.append(
            {
                "fit_u": float(point[0]),
                "fit_v": float(point[1]),
                "proj_u": float(q_best[0]),
                "proj_v": float(q_best[1]),
                "t_rad": t_best,
                "distance": float(np.linalg.norm(q_best - point)),
            }
        )

    return pd.DataFrame(rows)


def choose_direction_sign(t_values: Sequence[float]) -> int:
    t = np.asarray(t_values, dtype=float)
    if t.size < 2:
        return 1

    expected = 2.0 * math.pi / float(t.size)

    def forward_deltas(sign: float) -> np.ndarray:
        deltas = [
            float(np.mod(sign * (t[i] - t[i - 1]), 2.0 * math.pi))
            for i in range(1, len(t))
        ]
        deltas.append(float(np.mod(sign * (t[0] - t[-1]), 2.0 * math.pi)))
        return np.asarray(deltas, dtype=float)

    plus = forward_deltas(+1.0)
    minus = forward_deltas(-1.0)
    plus_score = float(np.mean((plus - expected) ** 2))
    minus_score = float(np.mean((minus - expected) ** 2))
    return -1 if minus_score < plus_score else 1


def build_path_lookup(
    ellipse: EllipseFit,
    direction_sign: int,
    grid_size: int,
) -> dict[str, object]:
    if grid_size < 512:
        raise ValueError("grid_size must be >= 512.")

    path_u_grid = np.linspace(0.0, 2.0 * math.pi, int(grid_size))
    actual_t_grid = float(direction_sign) * path_u_grid
    xy_grid = ellipse_points(ellipse, actual_t_grid)
    ds = np.sqrt(np.sum((xy_grid[1:] - xy_grid[:-1]) ** 2, axis=1))
    path_s_grid = np.r_[0.0, np.cumsum(ds)]
    circumference = float(path_s_grid[-1])

    def s_to_u(path_s: np.ndarray | float) -> np.ndarray:
        s_array = np.asarray(path_s, dtype=float)
        return np.interp(np.mod(s_array, circumference), path_s_grid, path_u_grid)

    def u_to_s(path_u: np.ndarray | float) -> np.ndarray:
        u_array = np.asarray(path_u, dtype=float)
        return np.interp(np.mod(u_array, 2.0 * math.pi), path_u_grid, path_s_grid)

    def point_from_s(path_s: np.ndarray | float) -> np.ndarray:
        u = s_to_u(path_s)
        t = float(direction_sign) * u
        return ellipse_points(ellipse, t)

    return {
        "path_u_grid": path_u_grid,
        "path_s_grid": path_s_grid,
        "xy_grid": xy_grid,
        "circumference": circumference,
        "s_to_u": s_to_u,
        "u_to_s": u_to_s,
        "point_from_s": point_from_s,
    }


def solve_equal_chord_polygon(
    start_s: float,
    n_points: int,
    point_from_s,
    circumference: float,
    max_chord: float,
) -> dict[str, object]:
    """
    Solve for a closed polygon on the ellipse with equal Euclidean neighbor spacing.

    The spacing constraint is the ambient 2D distance between consecutive
    points. Arc length along the ellipse between those points may vary.
    """
    if n_points < 3:
        raise ValueError("n_points must be >= 3.")

    nominal_step = circumference / float(n_points)

    def next_delta(current_s: float, chord: float, delta_guess: float) -> float | None:
        p0 = point_from_s(np.array([current_s], dtype=float))[0]

        def f(delta: float) -> float:
            p1 = point_from_s(np.array([current_s + delta], dtype=float))[0]
            return float(np.linalg.norm(p1 - p0) - chord)

        low = 1e-12
        high = max(delta_guess * 1.35, nominal_step * 0.5, 1e-6)
        value_high = f(high)
        while value_high < 0.0 and high < 0.5 * circumference:
            high *= 1.35
            value_high = f(high)
        if value_high < 0.0:
            return None
        return float(brentq(f, low, high, xtol=1e-10, rtol=1e-10, maxiter=100))

    def trace(chord: float) -> dict[str, object] | None:
        s_positions = [float(start_s)]
        deltas: list[float] = []
        current_s = float(start_s)
        guess = nominal_step
        for _ in range(n_points):
            delta = next_delta(current_s, chord, guess)
            if delta is None:
                return None
            deltas.append(delta)
            current_s += delta
            s_positions.append(current_s)
            guess = delta
        total_path = float(sum(deltas))
        return {
            "s_positions": np.asarray(s_positions[:-1], dtype=float),
            "deltas": np.asarray(deltas, dtype=float),
            "total_path": total_path,
        }

    arc_neighbor = point_from_s(np.array([start_s, start_s + nominal_step], dtype=float))
    low = 0.0
    high = float(np.linalg.norm(arc_neighbor[1] - arc_neighbor[0])) * 1.05
    trial = trace(high)
    expand_count = 0
    while (trial is None or float(trial["total_path"]) <= circumference) and high < max_chord:
        high *= 1.2
        expand_count += 1
        trial = trace(high)
        if expand_count > 60:
            break
    if trial is None or float(trial["total_path"]) <= circumference:
        raise RuntimeError("Failed to bracket equal-chord spacing solution on the ellipse.")

    best = trial
    for _ in range(48):
        chord = 0.5 * (low + high)
        trial = trace(chord)
        if trial is None or float(trial["total_path"]) > circumference:
            high = chord
            if trial is not None:
                best = trial
            continue
        low = chord

    chord = high
    final = trace(chord)
    if final is None:
        raise RuntimeError("Equal-chord spacing solver failed at the final chord length.")

    points = point_from_s(final["s_positions"])
    return {
        "start_s": float(start_s),
        "circumference": float(circumference),
        "chord_length": float(chord),
        "s_positions": final["s_positions"],
        "points": points,
        "step_arc_lengths": final["deltas"],
        "total_path": float(final["total_path"]),
    }


def mean_rgb_lookup(df: pd.DataFrame) -> dict[int, np.ndarray]:
    valid = df[df["valid_rgb"]].copy()
    if valid.empty:
        return {}
    grouped = (
        valid.groupby("id", as_index=False)[["r", "g", "b"]]
        .mean()
        .astype({"id": int})
    )
    return {
        int(row["id"]): np.clip(row[["r", "g", "b"]].to_numpy(dtype=float) / 255.0, 0.0, 1.0)
        for _, row in grouped.iterrows()
    }


def ids_to_text(ids: Sequence[int]) -> str:
    return ",".join(str(int(value)) for value in ids)


def plot_fit_samples(
    ellipse: EllipseFit,
    fit_vectors: pd.DataFrame,
    handle_table: pd.DataFrame,
    rgb_lookup: dict[int, np.ndarray],
    output_path: Path,
    show: bool,
) -> None:
    theta = np.linspace(0.0, 2.0 * math.pi, 1000)
    ellipse_xy = ellipse_points(ellipse, theta)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(ellipse_xy[:, 0], ellipse_xy[:, 1], color="black", linewidth=1.5, label="Fitted ellipse")

    for color_id, group in fit_vectors.groupby("color_id", sort=True):
        color = rgb_lookup.get(int(color_id), np.array([0.6, 0.6, 0.6]))
        ax.scatter(
            group["fit_u"],
            group["fit_v"],
            s=32,
            color=color,
            edgecolors="black",
            linewidths=0.25,
            alpha=0.75,
        )

    ax.scatter([0.0], [0.0], color="tab:red", marker="x", s=60, label="Grey origin")
    ax.scatter(
        handle_table["handle_fit_u"],
        handle_table["handle_fit_v"],
        color="tab:orange",
        edgecolors="black",
        s=70,
        linewidths=0.4,
        label="Handle means",
        zorder=4,
    )
    for _, row in handle_table.iterrows():
        ax.text(row["handle_fit_u"] + 0.5, row["handle_fit_v"] + 0.5, row["handle_label"], fontsize=9)

    ax.set_xlabel("Grey-relative U at normalized L*")
    ax.set_ylabel("Grey-relative V at normalized L*")
    ax.set_title("Historical target: fitted ellipse and measured vectors")
    ax.axhline(0.0, color="0.85", linewidth=0.8)
    ax.axvline(0.0, color="0.85", linewidth=0.8)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def plot_handle_alignment(
    ellipse: EllipseFit,
    fit_vectors: pd.DataFrame,
    handle_table: pd.DataFrame,
    final_target: pd.DataFrame,
    rgb_lookup: dict[int, np.ndarray],
    output_path: Path,
    show: bool,
) -> None:
    theta = np.linspace(0.0, 2.0 * math.pi, 1000)
    ellipse_xy = ellipse_points(ellipse, theta)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(ellipse_xy[:, 0], ellipse_xy[:, 1], color="black", linewidth=1.4, label="Fitted ellipse")

    for color_id, group in fit_vectors.groupby("color_id", sort=True):
        color = rgb_lookup.get(int(color_id), np.array([0.75, 0.75, 0.75]))
        ax.scatter(
            group["fit_u"],
            group["fit_v"],
            s=22,
            color=color,
            alpha=0.32,
            edgecolors="none",
        )

    handle_target = final_target[final_target["is_handle"]].copy()
    ax.scatter(
        handle_table["handle_fit_u"],
        handle_table["handle_fit_v"],
        color="tab:blue",
        s=70,
        label="Handle means",
        zorder=4,
    )
    ax.scatter(
        handle_target["fit_u"],
        handle_target["fit_v"],
        color="tab:orange",
        s=60,
        label="Aligned handles",
        zorder=5,
    )

    for (_, handle_row), (_, target_row) in zip(handle_table.iterrows(), handle_target.iterrows(), strict=True):
        ax.plot(
            [handle_row["handle_fit_u"], target_row["fit_u"]],
            [handle_row["handle_fit_v"], target_row["fit_v"]],
            color="0.55",
            linewidth=0.9,
        )
        ax.text(target_row["fit_u"] + 0.5, target_row["fit_v"] + 0.5, handle_row["handle_label"], fontsize=9)

    ax.set_xlabel("Grey-relative U at normalized L*")
    ax.set_ylabel("Grey-relative V at normalized L*")
    ax.set_title("Historical target: handle alignment")
    ax.axhline(0.0, color="0.85", linewidth=0.8)
    ax.axvline(0.0, color="0.85", linewidth=0.8)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def plot_final_target(
    ellipse: EllipseFit,
    final_target: pd.DataFrame,
    output_path: Path,
    show: bool,
) -> None:
    theta = np.linspace(0.0, 2.0 * math.pi, 1000)
    ellipse_xy = ellipse_points(ellipse, theta)
    hue_angles = np.mod(np.arctan2(final_target["fit_v"], final_target["fit_u"]), 2.0 * math.pi) / (2.0 * math.pi)
    colors = plt.cm.hsv(hue_angles.to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.plot(ellipse_xy[:, 0], ellipse_xy[:, 1], color="black", linewidth=1.3, label="Fitted ellipse")
    ax.scatter(final_target["fit_u"], final_target["fit_v"], c=colors, s=44, edgecolors="black", linewidths=0.25)

    handles = final_target[final_target["is_handle"]]
    ax.scatter(
        handles["fit_u"],
        handles["fit_v"],
        facecolors="none",
        edgecolors="tab:orange",
        linewidths=1.8,
        s=100,
        label="Handles",
        zorder=5,
    )

    for _, row in final_target.iterrows():
        ax.text(row["fit_u"] + 0.35, row["fit_v"] + 0.35, str(int(row["target_id"])), fontsize=7)

    ax.scatter([0.0], [0.0], color="tab:red", marker="x", s=60, label="Grey origin")
    ax.set_xlabel("Grey-relative U at normalized L*")
    ax.set_ylabel("Grey-relative V at normalized L*")
    ax.set_title("Historical target: final output colors")
    ax.axhline(0.0, color="0.85", linewidth=0.8)
    ax.axvline(0.0, color="0.85", linewidth=0.8)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construct a historical grey-relative CIE-LUV target from measured XYZ or raw spectra files."
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        required=True,
        help=(
            "One or more input files. Each may be raw spectra "
            "(rep,id,r,g,b,nm,power) or measured XYZ (rep,id,r,g,b,X,Y,Z)."
        ),
    )
    parser.add_argument(
        "--cmf",
        default="1931",
        help="Color matching function for spectral inputs. Supported values include 1931 and judd. Default: 1931",
    )
    parser.add_argument(
        "--cmf-file",
        default=None,
        help="Optional custom CMF file path for spectral inputs.",
    )
    parser.add_argument("--white-id", type=int, default=0, help="Measurement ID for the white point.")
    parser.add_argument("--black-id", type=int, default=1, help="Measurement ID for the black point.")
    parser.add_argument("--grey-id", type=int, default=2, help="Measurement ID for the grey point.")
    parser.add_argument(
        "--fit-color-ids",
        default=None,
        help="Comma-separated measured color IDs or inclusive ranges such as '3-8,12'. "
        "Defaults to all IDs except white, black, and grey.",
    )
    parser.add_argument(
        "--handle-groups",
        default=None,
        help="Semicolon-separated handle groups. Each group is a comma-separated ID list or range. "
        "Example: '3,9,15;4,10,16;5,11,17'. Order defines handle order and target direction.",
    )
    parser.add_argument(
        "--handle-labels",
        default=None,
        help="Comma-separated labels matching --handle-groups. Defaults to the group IDs themselves.",
    )
    parser.add_argument(
        "--n-output-colors",
        type=int,
        default=36,
        help="Number of final output colors on the ellipse. Neighboring colors are constrained "
        "to have equal Euclidean distance in the fit plane.",
    )
    parser.add_argument(
        "--uv-normalization-lstar",
        type=float,
        default=100.0,
        help="L* used to normalize grey-relative chromatic vectors for fitting.",
    )
    parser.add_argument(
        "--circularity-weight",
        type=float,
        default=0.1,
        help="Regularization weight that biases the fitted ellipse toward a circle.",
    )
    parser.add_argument(
        "--ellipse-max-iter",
        type=int,
        default=4000,
        help="Maximum optimizer iterations for the ellipse fit.",
    )
    parser.add_argument(
        "--arc-grid-size",
        type=int,
        default=5001,
        help="Number of dense samples used to parameterize ellipse arc length internally. "
        "Final target spacing is not arc-uniform.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for this calibration stage's outputs.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    input_paths = [Path(path) for path in args.input_files]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")

    output_root = Path(args.output_root)
    plots_dir = output_root / "plots"
    data_dir = output_root / "data"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    input_file_modes: list[dict[str, object]] = []
    cmf_key: str | None = None
    cmf_path: Path | None = None
    cmf_wavelengths: np.ndarray | None = None
    cmf_values: np.ndarray | None = None
    for tablet_index, path in enumerate(input_paths):
        table = read_table_auto(path)
        if SPD_COLUMNS.issubset(table.columns):
            if cmf_key is None or cmf_path is None or cmf_wavelengths is None or cmf_values is None:
                cmf_key, cmf_path, cmf_wavelengths, cmf_values = load_cmf(args.cmf, args.cmf_file)
            xyz_df = convert_spd_df_to_xyz(
                table,
                cmf_wavelengths=cmf_wavelengths,
                cmf_values=cmf_values,
                source_file=path.name,
            )
            input_mode = "spd"
        elif XYZ_COLUMNS.issubset(table.columns):
            xyz_df = table.copy()
            input_mode = "xyz"
        else:
            raise ValueError(
                f"{path} does not contain a supported input schema. "
                f"Need either SPD columns {sorted(SPD_COLUMNS)} or XYZ columns {sorted(XYZ_COLUMNS)}."
            )

        frames.append(normalize_and_validate(xyz_df, path.name, tablet_index))
        input_file_modes.append(
            {
                "tablet_index": int(tablet_index),
                "input_file": str(path.resolve()),
                "input_mode": input_mode,
            }
        )
    all_rows = pd.concat(frames, ignore_index=True)

    excluded_ids = {int(args.white_id), int(args.black_id), int(args.grey_id)}
    detected_color_ids = sorted(int(value) for value in all_rows["id"].unique().tolist() if int(value) not in excluded_ids)
    fit_color_ids = parse_id_spec(args.fit_color_ids) if args.fit_color_ids else detected_color_ids
    fit_color_ids = [int(value) for value in fit_color_ids if int(value) not in excluded_ids]
    fit_color_ids = list(dict.fromkeys(fit_color_ids))
    if not fit_color_ids:
        raise RuntimeError("No fit color IDs were selected.")

    explicit_handle_groups_provided = bool(args.handle_groups and str(args.handle_groups).strip())
    if explicit_handle_groups_provided:
        handle_groups = parse_handle_groups(args.handle_groups)
        handle_groups = [[int(value) for value in group] for group in handle_groups]
        handle_labels = parse_labels(args.handle_labels)
        if handle_labels and len(handle_labels) != len(handle_groups):
            raise ValueError("--handle-labels must have the same number of entries as --handle-groups.")
        if not handle_labels:
            handle_labels = [ids_to_text(group) for group in handle_groups]
        handle_ids_union = {value for group in handle_groups for value in group}
        missing_handle_ids = sorted(handle_ids_union - set(int(value) for value in all_rows["id"].unique().tolist()))
        if missing_handle_ids:
            raise RuntimeError(f"Handle group IDs were not found in the inputs: {missing_handle_ids}")
    else:
        handle_groups = []
        handle_labels = []
        handle_ids_union = set()
        missing_handle_ids = []

    tablet_summary_rows: list[dict[str, object]] = []
    vector_rows: list[dict[str, object]] = []

    relevant_color_ids = set(fit_color_ids) | handle_ids_union

    for (tablet_index, source_file), source_df in all_rows.groupby(["tablet_index", "source_file"], sort=False):
        rep_groups = list(source_df.groupby("rep", sort=False))
        valid_rep_count = 0

        for rep, rep_df in rep_groups:
            white_rows = rep_df[(rep_df["id"] == args.white_id) & rep_df["valid_xyz"]]
            grey_rows = rep_df[(rep_df["id"] == args.grey_id) & rep_df["valid_xyz"]]
            black_rows = rep_df[(rep_df["id"] == args.black_id) & rep_df["valid_xyz"]]

            if white_rows.empty or grey_rows.empty:
                tablet_summary_rows.append(
                    {
                        "tablet_index": int(tablet_index),
                        "source_file": source_file,
                        "rep": int(rep),
                        "has_valid_white": bool(not white_rows.empty),
                        "has_valid_grey": bool(not grey_rows.empty),
                        "has_valid_black": bool(not black_rows.empty),
                        "white_X": np.nan,
                        "white_Y": np.nan,
                        "white_Z": np.nan,
                        "grey_X": np.nan,
                        "grey_Y": np.nan,
                        "grey_Z": np.nan,
                        "black_X": np.nan,
                        "black_Y": np.nan,
                        "black_Z": np.nan,
                        "n_valid_vectors": 0,
                    }
                )
                continue

            valid_rep_count += 1
            white_xyz = white_rows[["X", "Y", "Z"]].mean().to_numpy(dtype=float)
            grey_xyz = grey_rows[["X", "Y", "Z"]].mean().to_numpy(dtype=float)
            black_xyz = (
                black_rows[["X", "Y", "Z"]].mean().to_numpy(dtype=float)
                if not black_rows.empty
                else np.array([np.nan, np.nan, np.nan], dtype=float)
            )

            grey_luv = xyz_to_cie_luv(grey_xyz.reshape(1, 3), white_xyz)[0]
            grey_up, grey_vp = xyz_to_upvp(grey_xyz.reshape(1, 3))
            n_valid_vectors = 0

            for row in rep_df.itertuples(index=False):
                color_id = int(row.id)
                if color_id not in relevant_color_ids:
                    continue
                if not bool(row.valid_xyz):
                    continue

                color_xyz = np.array([row.X, row.Y, row.Z], dtype=float).reshape(1, 3)
                color_luv = xyz_to_cie_luv(color_xyz, white_xyz)[0]
                color_up, color_vp = xyz_to_upvp(color_xyz)

                dup = float(color_up[0] - grey_up[0])
                dvp = float(color_vp[0] - grey_vp[0])
                fit_u = float(13.0 * args.uv_normalization_lstar * dup)
                fit_v = float(13.0 * args.uv_normalization_lstar * dvp)
                delta_l = float(color_luv[0] - grey_luv[0])
                delta_u = float(color_luv[1] - grey_luv[1])
                delta_v = float(color_luv[2] - grey_luv[2])

                vector_rows.append(
                    {
                        "tablet_index": int(tablet_index),
                        "source_file": source_file,
                        "rep": int(rep),
                        "color_id": color_id,
                        "r": float(row.r),
                        "g": float(row.g),
                        "b": float(row.b),
                        "X": float(row.X),
                        "Y": float(row.Y),
                        "Z": float(row.Z),
                        "white_X": float(white_xyz[0]),
                        "white_Y": float(white_xyz[1]),
                        "white_Z": float(white_xyz[2]),
                        "grey_X": float(grey_xyz[0]),
                        "grey_Y": float(grey_xyz[1]),
                        "grey_Z": float(grey_xyz[2]),
                        "color_L": float(color_luv[0]),
                        "color_u": float(color_luv[1]),
                        "color_v": float(color_luv[2]),
                        "grey_L": float(grey_luv[0]),
                        "grey_u": float(grey_luv[1]),
                        "grey_v": float(grey_luv[2]),
                        "delta_L": delta_l,
                        "delta_u": delta_u,
                        "delta_v": delta_v,
                        "color_up": float(color_up[0]),
                        "color_vp": float(color_vp[0]),
                        "grey_up": float(grey_up[0]),
                        "grey_vp": float(grey_vp[0]),
                        "dup": dup,
                        "dvp": dvp,
                        "fit_u": fit_u,
                        "fit_v": fit_v,
                    }
                )
                n_valid_vectors += 1

            tablet_summary_rows.append(
                {
                    "tablet_index": int(tablet_index),
                    "source_file": source_file,
                    "rep": int(rep),
                    "has_valid_white": True,
                    "has_valid_grey": True,
                    "has_valid_black": bool(not black_rows.empty),
                    "white_X": float(white_xyz[0]),
                    "white_Y": float(white_xyz[1]),
                    "white_Z": float(white_xyz[2]),
                    "grey_X": float(grey_xyz[0]),
                    "grey_Y": float(grey_xyz[1]),
                    "grey_Z": float(grey_xyz[2]),
                    "black_X": float(black_xyz[0]),
                    "black_Y": float(black_xyz[1]),
                    "black_Z": float(black_xyz[2]),
                    "n_valid_vectors": int(n_valid_vectors),
                }
            )

        if valid_rep_count == 0:
            raise RuntimeError(f"{source_file} did not contain any rep with both valid white and grey rows.")

    tablet_summary = pd.DataFrame(tablet_summary_rows)
    vector_df = pd.DataFrame(vector_rows)
    if vector_df.empty:
        raise RuntimeError("No valid color vectors were constructed from the inputs.")

    valid_grey_rows = tablet_summary[
        tablet_summary["has_valid_white"]
        & tablet_summary["has_valid_grey"]
        & np.isfinite(tablet_summary[["grey_X", "grey_Y", "grey_Z"]]).all(axis=1)
    ].copy()
    if valid_grey_rows.empty:
        raise RuntimeError("No valid grey-point XYZ measurements were available for averaging.")

    per_tablet_grey_xyz = (
        valid_grey_rows.groupby(["tablet_index", "source_file"], as_index=False)
        .agg(
            n_valid_reps=("rep", "size"),
            grey_X=("grey_X", "mean"),
            grey_Y=("grey_Y", "mean"),
            grey_Z=("grey_Z", "mean"),
        )
        .sort_values(["tablet_index", "source_file"])
        .reset_index(drop=True)
    )
    per_tablet_grey_xyz.insert(0, "scope", "tablet_mean")

    overall_grey_xyz = pd.DataFrame(
        [
            {
                "scope": "overall_tablet_mean",
                "tablet_index": np.nan,
                "source_file": "ALL_TABLETS",
                "n_valid_reps": int(per_tablet_grey_xyz["n_valid_reps"].sum()),
                "grey_X": float(per_tablet_grey_xyz["grey_X"].mean()),
                "grey_Y": float(per_tablet_grey_xyz["grey_Y"].mean()),
                "grey_Z": float(per_tablet_grey_xyz["grey_Z"].mean()),
            }
        ]
    )
    average_grey_xyz = pd.concat([per_tablet_grey_xyz, overall_grey_xyz], ignore_index=True)

    fit_vectors = vector_df[vector_df["color_id"].isin(fit_color_ids)].copy()
    if fit_vectors.empty:
        raise RuntimeError("No valid fit vectors remain after applying --fit-color-ids.")

    fit_counts = fit_vectors.groupby("color_id", as_index=False).size().rename(columns={"size": "n_samples"})
    missing_fit_ids = sorted(set(fit_color_ids) - set(int(value) for value in fit_counts["color_id"].tolist()))
    present_fit_ids = sorted(int(value) for value in fit_counts["color_id"].tolist())

    if not explicit_handle_groups_provided:
        handle_groups = [[color_id] for color_id in present_fit_ids]
        handle_labels = [ids_to_text(group) for group in handle_groups]

    n_output_colors = int(args.n_output_colors)
    n_handles = len(handle_groups)
    if n_output_colors < 3:
        raise ValueError("--n-output-colors must be >= 3.")
    if n_handles < 2:
        raise ValueError("At least two handle groups are required.")
    if n_output_colors % n_handles != 0:
        if not explicit_handle_groups_provided and missing_fit_ids:
            raise ValueError(
                "Default handle groups are built from fit-color IDs with valid vectors. "
                f"After filtering, these IDs were missing entirely: {missing_fit_ids}. "
                f"That leaves {n_handles} default handles, and --n-output-colors={n_output_colors} "
                "is not divisible by that count. For the historical tablet data, pass explicit "
                "--handle-groups such as "
                "'3,9,15;4,10,16;5,11,17;6,12,18;7,13,19;8,14,20'."
            )
        raise ValueError(
            "--n-output-colors must be divisible by the number of handle groups so handles are evenly spaced."
        )
    colors_per_segment = n_output_colors // n_handles

    handle_rows: list[dict[str, object]] = []
    for handle_index, (label, group_ids) in enumerate(zip(handle_labels, handle_groups, strict=True)):
        group_fit_ids = sorted(set(int(value) for value in group_ids) & set(fit_color_ids))
        group_vectors = vector_df[vector_df["color_id"].isin(group_fit_ids)].copy()
        if group_vectors.empty:
            raise RuntimeError(f"Handle group '{label}' has no valid vectors.")
        handle_rows.append(
            {
                "handle_index": int(handle_index),
                "handle_label": label,
                "handle_group_ids": ids_to_text(group_ids),
                "fit_group_ids": ids_to_text(group_fit_ids),
                "n_samples": int(len(group_vectors)),
                "handle_fit_u": float(group_vectors["fit_u"].mean()),
                "handle_fit_v": float(group_vectors["fit_v"].mean()),
                "handle_delta_L_mean": float(group_vectors["delta_L"].mean()),
                "handle_delta_u_mean": float(group_vectors["delta_u"].mean()),
                "handle_delta_v_mean": float(group_vectors["delta_v"].mean()),
            }
        )
    handle_table = pd.DataFrame(handle_rows)

    ellipse, cluster_fit_table = fit_origin_centered_ellipse(
        handle_table[["handle_fit_u", "handle_fit_v"]].to_numpy(dtype=float),
        circularity_weight=float(args.circularity_weight),
        max_iter=int(args.ellipse_max_iter),
        weights=handle_table["n_samples"].to_numpy(dtype=float),
    )
    cluster_fit_table = cluster_fit_table.rename(
        columns={
            "fit_u": "cluster_fit_u",
            "fit_v": "cluster_fit_v",
            "weight": "cluster_weight",
            "rho": "cluster_rho",
            "ellipse_residual": "cluster_ellipse_residual",
        }
    )
    handle_table = pd.concat([handle_table.reset_index(drop=True), cluster_fit_table], axis=1)

    fit_vectors = fit_vectors.reset_index(drop=True)
    fit_vectors = pd.concat(
        [fit_vectors, ellipse_residuals_for_points(fit_vectors[["fit_u", "fit_v"]].to_numpy(dtype=float), ellipse)],
        axis=1,
    )

    handle_projection = project_points_to_ellipse(
        handle_table[["handle_fit_u", "handle_fit_v"]].to_numpy(dtype=float),
        ellipse=ellipse,
    )
    handle_table = pd.concat([handle_table, handle_projection], axis=1)

    direction_sign = choose_direction_sign(handle_table["t_rad"].to_numpy(dtype=float))
    path_lookup = build_path_lookup(
        ellipse=ellipse,
        direction_sign=direction_sign,
        grid_size=int(args.arc_grid_size),
    )
    circumference = float(path_lookup["circumference"])
    u_to_s = path_lookup["u_to_s"]
    point_from_s = path_lookup["point_from_s"]

    t_raw = handle_table["t_rad"].to_numpy(dtype=float)
    u_raw = np.mod(direction_sign * t_raw, 2.0 * math.pi)
    s_raw = u_to_s(u_raw)
    start_s_seed = float(s_raw[0])
    max_chord = 2.0 * float(ellipse.axis_a) * 1.01

    def alignment_objective(start_s: float) -> float:
        solved = solve_equal_chord_polygon(
            start_s=float(start_s),
            n_points=n_output_colors,
            point_from_s=point_from_s,
            circumference=circumference,
            max_chord=max_chord,
        )
        handle_indices = np.arange(n_handles, dtype=int) * colors_per_segment
        handle_points = solved["points"][handle_indices]
        deltas = handle_points - handle_table[["handle_fit_u", "handle_fit_v"]].to_numpy(dtype=float)
        return float(np.mean(np.sum(deltas**2, axis=1)))

    search_half_width = circumference / float(n_output_colors)
    start_opt = minimize_scalar(
        alignment_objective,
        bounds=(start_s_seed - search_half_width, start_s_seed + search_half_width),
        method="bounded",
        options={"maxiter": 80},
    )

    solved = solve_equal_chord_polygon(
        start_s=float(start_opt.x),
        n_points=n_output_colors,
        point_from_s=point_from_s,
        circumference=circumference,
        max_chord=max_chord,
    )

    handle_indices = np.arange(n_handles, dtype=int) * colors_per_segment
    handle_positions = solved["points"][handle_indices]
    handle_s = solved["s_positions"][handle_indices]
    handle_table["target_path_s"] = handle_s
    handle_table["target_fit_u"] = handle_positions[:, 0]
    handle_table["target_fit_v"] = handle_positions[:, 1]
    handle_table["alignment_distance"] = np.sqrt(
        (handle_table["target_fit_u"] - handle_table["handle_fit_u"]) ** 2
        + (handle_table["target_fit_v"] - handle_table["handle_fit_v"]) ** 2
    )

    adjacency = np.sqrt(
        np.sum((np.roll(solved["points"], -1, axis=0) - solved["points"]) ** 2, axis=1)
    )

    final_rows: list[dict[str, object]] = []
    for target_id in range(n_output_colors):
        segment_idx = target_id // colors_per_segment
        handle_index = segment_idx
        point = solved["points"][target_id]
        final_rows.append(
            {
                "target_id": int(target_id),
                "segment_idx": int(segment_idx),
                "handle_index": int(handle_index),
                "handle_label": handle_labels[handle_index],
                "source_handle_group_ids": ids_to_text(handle_groups[handle_index]),
                "next_handle_label": handle_labels[(handle_index + 1) % n_handles],
                "next_handle_group_ids": ids_to_text(handle_groups[(handle_index + 1) % n_handles]),
                "segment_position": int(target_id % colors_per_segment),
                "is_handle": bool(target_id % colors_per_segment == 0),
                "path_s": float(solved["s_positions"][target_id]),
                "fit_u": float(point[0]),
                "fit_v": float(point[1]),
                "adjacent_arc_length": float(solved["step_arc_lengths"][target_id]),
                "adjacent_chord_length": float(adjacency[target_id]),
            }
        )
    final_target = pd.DataFrame(final_rows)

    compact_rows: list[dict[str, object]] = []
    for row in final_target.itertuples(index=False):
        if bool(row.is_handle):
            description = f"{row.handle_label} handle"
        else:
            description = (
                f"between {row.handle_label} and {row.next_handle_label}, "
                f"step {int(row.segment_position)} of {colors_per_segment - 1}"
            )
        compact_rows.append(
            {
                "id": int(row.target_id),
                "description": description,
                # This stage defines the hue ring in grey-relative U,V only.
                "delta_L": 0.0,
                "delta_U": float(row.fit_u),
                "delta_V": float(row.fit_v),
            }
        )
    compact_displacements = pd.DataFrame(compact_rows)

    fit_color_summary = (
        fit_vectors.groupby("color_id", as_index=False)
        .agg(
            n_samples=("color_id", "size"),
            fit_u_mean=("fit_u", "mean"),
            fit_v_mean=("fit_v", "mean"),
            delta_L_mean=("delta_L", "mean"),
            delta_u_mean=("delta_u", "mean"),
            delta_v_mean=("delta_v", "mean"),
            rho_mean=("rho", "mean"),
            ellipse_residual_mean=("ellipse_residual", "mean"),
        )
        .sort_values("color_id")
        .reset_index(drop=True)
    )

    rgb_lookup = mean_rgb_lookup(all_rows)

    plot_fit_samples(
        ellipse=ellipse,
        fit_vectors=fit_vectors,
        handle_table=handle_table,
        rgb_lookup=rgb_lookup,
        output_path=plots_dir / "01_fit_ellipse_and_samples.png",
        show=bool(args.show_plots),
    )
    plot_handle_alignment(
        ellipse=ellipse,
        fit_vectors=fit_vectors,
        handle_table=handle_table,
        final_target=final_target,
        rgb_lookup=rgb_lookup,
        output_path=plots_dir / "02_handle_alignment.png",
        show=bool(args.show_plots),
    )
    plot_final_target(
        ellipse=ellipse,
        final_target=final_target,
        output_path=plots_dir / "03_final_target_colors.png",
        show=bool(args.show_plots),
    )

    tablet_summary.to_csv(data_dir / "tablet_rep_summary.csv", index=False)
    average_grey_xyz.to_csv(data_dir / "average_grey_xyz.csv", index=False)
    vector_df.to_csv(data_dir / "vector_samples.csv", index=False)
    fit_vectors.to_csv(data_dir / "fit_vectors_with_residuals.csv", index=False)
    fit_color_summary.to_csv(data_dir / "fit_color_summary.csv", index=False)
    handle_table.to_csv(data_dir / "handle_alignment.csv", index=False)
    handle_table.to_csv(data_dir / "fit_clusters.csv", index=False)
    final_target.to_csv(data_dir / "historical_cieluv_target.csv", index=False)
    compact_displacements.to_csv(data_dir / "compact_LUV_displacenments.csv", index=False)

    dense_curve = pd.DataFrame(
        {
            "path_s": path_lookup["path_s_grid"],
            "fit_u": path_lookup["xy_grid"][:, 0],
            "fit_v": path_lookup["xy_grid"][:, 1],
        }
    )
    dense_curve.to_csv(data_dir / "ellipse_reference_curve.csv", index=False)

    ellipse_json = {
        "fit_space": {
            "description": "Grey-relative chromatic vectors in CIE-LUV units normalized to a fixed L*.",
            "normalized_lstar": float(args.uv_normalization_lstar),
            "u_formula": "13 * L_norm * (u'_color - u'_grey)",
            "v_formula": "13 * L_norm * (v'_color - v'_grey)",
            "ellipse_fit_basis": "Handle clusters are averaged across their grouped lightness levels, then weighted by valid sample count.",
        },
        "ellipse": {
            "center_u": ellipse.center_u,
            "center_v": ellipse.center_v,
            "axis_a": ellipse.axis_a,
            "axis_b": ellipse.axis_b,
            "axis_ratio": ellipse.axis_ratio,
            "angle_rad": ellipse.angle_rad,
            "rmse": ellipse.rmse,
            "objective": ellipse.objective,
            "iterations": ellipse.iterations,
            "converged": ellipse.converged,
            "parametric_equation": {
                "u_t": "a*cos(t)*cos(phi) - b*sin(t)*sin(phi)",
                "v_t": "a*cos(t)*sin(phi) + b*sin(t)*cos(phi)",
            },
        },
        "alignment": {
            "spacing_definition": (
                "Adjacent target colors are constrained to have equal Euclidean distance "
                "in the ambient 2D fit plane (equal chord length). Ellipse arc lengths "
                "between neighbors are not constrained to be equal."
            ),
            "direction_sign": int(direction_sign),
            "circumference": float(circumference),
            "colors_per_segment": int(colors_per_segment),
            "n_output_colors": int(n_output_colors),
            "n_handles": int(n_handles),
            "equal_chord_length": float(solved["chord_length"]),
            "start_path_s": float(solved["start_s"]),
            "alignment_objective": float(start_opt.fun),
            "adjacent_arc_length_min": float(np.min(solved["step_arc_lengths"])),
            "adjacent_arc_length_max": float(np.max(solved["step_arc_lengths"])),
            "adjacent_arc_length_mean": float(np.mean(solved["step_arc_lengths"])),
            "adjacent_chord_length_min": float(np.min(adjacency)),
            "adjacent_chord_length_max": float(np.max(adjacency)),
            "adjacent_chord_length_mean": float(np.mean(adjacency)),
        },
        "ids": {
            "white_id": int(args.white_id),
            "black_id": int(args.black_id),
            "grey_id": int(args.grey_id),
            "fit_color_ids": fit_color_ids,
            "fit_color_ids_skipped_as_missing": missing_fit_ids,
            "handle_groups": handle_groups,
            "handle_labels": handle_labels,
        },
        "inputs": [str(path) for path in input_paths],
        "input_modes": input_file_modes,
        "cmf": cmf_key,
        "cmf_file": str(cmf_path.resolve()) if cmf_path is not None else None,
    }
    with (data_dir / "ellipse_fit.json").open("w", encoding="utf-8") as handle:
        json.dump(ellipse_json, handle, indent=2)

    run_config = {
        "input_files": [str(path) for path in input_paths],
        "input_modes": input_file_modes,
        "cmf": cmf_key,
        "cmf_file": str(cmf_path.resolve()) if cmf_path is not None else None,
        "white_id": int(args.white_id),
        "black_id": int(args.black_id),
        "grey_id": int(args.grey_id),
        "fit_color_ids": fit_color_ids,
        "fit_color_ids_skipped_as_missing": missing_fit_ids,
        "handle_groups": handle_groups,
        "handle_labels": handle_labels,
        "n_output_colors": int(n_output_colors),
        "uv_normalization_lstar": float(args.uv_normalization_lstar),
        "circularity_weight": float(args.circularity_weight),
        "ellipse_max_iter": int(args.ellipse_max_iter),
        "arc_grid_size": int(args.arc_grid_size),
        "output_root": str(output_root),
    }
    with (data_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    print(f"Saved outputs to: {output_root}")
    if cmf_key is not None and cmf_path is not None:
        print("CMF:", cmf_key, "from", cmf_path)
    print(f"Valid vector samples: {len(vector_df)}")
    print(f"Fit sample vectors retained: {len(fit_vectors)}")
    print(f"Fit clusters used: {len(handle_table)}")
    print(
        "Average grey XYZ across tablets: "
        f"X={overall_grey_xyz.iloc[0]['grey_X']:.6f}, "
        f"Y={overall_grey_xyz.iloc[0]['grey_Y']:.6f}, "
        f"Z={overall_grey_xyz.iloc[0]['grey_Z']:.6f}"
    )
    if missing_fit_ids:
        print(f"Requested fit IDs skipped because they were entirely missing: {missing_fit_ids}")
    print(f"Handle groups: {n_handles}")
    print(f"Final target colors: {n_output_colors}")
    print(
        "Equal Euclidean neighbor distance summary: "
        f"min={np.min(adjacency):.6f}, mean={np.mean(adjacency):.6f}, max={np.max(adjacency):.6f}"
    )


if __name__ == "__main__":
    main()
