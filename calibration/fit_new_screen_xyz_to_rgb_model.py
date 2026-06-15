#!/usr/bin/env python3
"""
Fit an XYZ -> RGB screen model for the new display.

This stage is self-contained inside `calibration/` and writes all outputs under:

    calibration/outputs/02_fit_screen_model/

Input files may be either:
- raw spectra tables with columns: rep,id,r,g,b,nm,power
- precomputed XYZ tables with columns: rep,id,r,g,b,X,Y,Z

Raw spectra are converted to XYZ using the requested color matching functions.
For this stage the default is true XYZ 1931 (`ciexyz31.txt`), not the older
notebook's Judd-only helper.

Train/test splitting is ID-disjoint: all reps for a given color ID go entirely
to either the training set or the test set. Within each split, every rep is
still treated as a separate fitting data point.
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
from scipy.interpolate import CubicSpline

try:
    from .rgb_to_xyz_mlp import RGBToXYZMLP
    from .xyz_to_rgb_mlp import XYZToRGBMLP
except ImportError:  # pragma: no cover
    from rgb_to_xyz_mlp import RGBToXYZMLP
    from xyz_to_rgb_mlp import XYZToRGBMLP


DEFAULT_INPUT_FILE = (
    Path(__file__).resolve().parents[1] / "input_files" / "color_battery_amoled_B.csv"
)
DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parent / "outputs" / "02_fit_screen_model"
)
DEFAULT_CMF_1931 = (
    Path(__file__).resolve().parents[1] / "color_matching_functions" / "ciexyz31.txt"
)
DEFAULT_CMF_JUDD = (
    Path(__file__).resolve().parents[1] / "color_matching_functions" / "ciexyzj.txt"
)

SPD_COLUMNS = {"rep", "id", "r", "g", "b", "nm", "power"}
XYZ_COLUMNS = {"rep", "id", "r", "g", "b", "X", "Y", "Z"}


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
            cmf_wavelengths, cmf_values, wavelengths, power
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


def normalize_xyz_table(df: pd.DataFrame, source_file: str) -> pd.DataFrame:
    missing = XYZ_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{source_file} is missing required XYZ columns: {sorted(missing)}")

    work = df.copy()
    for column in ["rep", "id", "r", "g", "b", "X", "Y", "Z"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["rep", "id", "r", "g", "b", "X", "Y", "Z"]).copy()
    work["rep"] = work["rep"].astype(int)
    work["id"] = work["id"].astype(int)
    work["source_file"] = source_file
    return work


def clean_xyz_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    work = df.copy()
    work["valid_rgb"] = np.isfinite(work[["r", "g", "b"]]).all(axis=1)
    work["valid_rgb"] &= (work[["r", "g", "b"]] >= 0.0).all(axis=1)
    work["valid_rgb"] &= (work[["r", "g", "b"]] <= 255.0).all(axis=1)

    work["valid_xyz"] = np.isfinite(work[["X", "Y", "Z"]]).all(axis=1)
    work["valid_xyz"] &= ~(work[["X", "Y", "Z"]] == -1.0).any(axis=1)

    clean = work[work["valid_rgb"] & work["valid_xyz"]].copy().reset_index(drop=True)
    clean["r"] = clean["r"].astype(int)
    clean["g"] = clean["g"].astype(int)
    clean["b"] = clean["b"].astype(int)

    summary = {
        "n_rows_raw": int(len(work)),
        "n_rows_clean": int(len(clean)),
        "dropped_invalid_rgb": int((~work["valid_rgb"]).sum()),
        "dropped_invalid_xyz": int((~work["valid_xyz"]).sum()),
    }
    return clean, summary


def split_train_test_ids(
    df: pd.DataFrame,
    train_fraction: float,
    guaranteed_train_ids: list[int],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1.")
    if len(df) < 2:
        raise ValueError("Need at least two rows to create train/test splits.")

    rng = np.random.default_rng(int(seed))
    unique_ids = sorted(int(value) for value in df["id"].dropna().unique().tolist())
    if len(unique_ids) < 2:
        raise ValueError("Need at least two unique IDs to create an ID-disjoint split.")
    unique_id_set = set(unique_ids)

    desired_n_train_ids = int(math.floor(float(train_fraction) * len(unique_ids)))
    desired_n_train_ids = max(1, min(desired_n_train_ids, len(unique_ids) - 1))

    guaranteed_present_ids: list[int] = []
    guaranteed_status: list[dict[str, Any]] = []
    for color_id in guaranteed_train_ids:
        color_id = int(color_id)
        if color_id not in unique_id_set:
            guaranteed_status.append(
                {
                    "id": color_id,
                    "present": False,
                }
            )
            continue
        guaranteed_present_ids.append(color_id)
        guaranteed_status.append(
            {
                "id": color_id,
                "present": True,
            }
        )

    guaranteed_present_ids = sorted(set(guaranteed_present_ids))
    if len(guaranteed_present_ids) >= len(unique_ids):
        raise RuntimeError(
            "Guaranteed training IDs consume all available IDs; cannot create a held-out test set."
        )

    desired_n_train_ids = max(desired_n_train_ids, len(guaranteed_present_ids))
    desired_n_train_ids = min(desired_n_train_ids, len(unique_ids) - 1)

    remaining_ids = np.asarray(
        [color_id for color_id in unique_ids if color_id not in set(guaranteed_present_ids)],
        dtype=int,
    )
    rng.shuffle(remaining_ids)

    extra_needed = desired_n_train_ids - len(guaranteed_present_ids)
    chosen_train_ids = list(guaranteed_present_ids)
    if extra_needed > 0:
        chosen_train_ids.extend(remaining_ids[:extra_needed].tolist())

    train_ids = sorted(int(x) for x in chosen_train_ids)
    train_id_set = set(train_ids)
    test_ids = sorted(int(x) for x in unique_ids if x not in train_id_set)
    if len(test_ids) == 0:
        raise RuntimeError("Train/test split produced an empty test set.")

    train_df = df[df["id"].isin(train_ids)].copy().reset_index(drop=True)
    test_df = df[df["id"].isin(test_ids)].copy().reset_index(drop=True)

    split_summary = {
        "seed": int(seed),
        "train_fraction": float(train_fraction),
        "split_mode": "id_disjoint",
        "n_ids_total": int(len(unique_ids)),
        "n_ids_train": int(len(train_ids)),
        "n_ids_test": int(len(test_ids)),
        "train_ids": train_ids,
        "test_ids": test_ids,
        "n_rows_total": int(len(df)),
        "n_rows_train": int(len(train_df)),
        "n_rows_test": int(len(test_df)),
        "guaranteed_train_ids": [int(x) for x in guaranteed_train_ids],
        "guaranteed_train_id_status": guaranteed_status,
    }
    return train_df, test_df, split_summary


def predict_dataframe(
    model: XYZToRGBMLP,
    df: pd.DataFrame,
    split_name: str,
    *,
    warn: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    xyz = df[["X", "Y", "Z"]].to_numpy(dtype=float)
    bundle = model.predict_rgb(xyz, warn=warn, clip=True)
    preds = df.copy()
    preds["split"] = split_name
    preds["pred_scaled_r"] = bundle.scaled_rgb[:, 0]
    preds["pred_scaled_g"] = bundle.scaled_rgb[:, 1]
    preds["pred_scaled_b"] = bundle.scaled_rgb[:, 2]
    preds["pred_r_float"] = bundle.rgb_float[:, 0]
    preds["pred_g_float"] = bundle.rgb_float[:, 1]
    preds["pred_b_float"] = bundle.rgb_float[:, 2]
    preds["pred_r_int"] = bundle.rgb_int[:, 0]
    preds["pred_g_int"] = bundle.rgb_int[:, 1]
    preds["pred_b_int"] = bundle.rgb_int[:, 2]
    preds["pred_row_out_of_range"] = np.any(bundle.out_of_range_mask, axis=1)
    preds["pred_out_of_range_channels"] = np.sum(bundle.out_of_range_mask, axis=1).astype(int)
    preds["abs_err_r"] = np.abs(preds["pred_r_int"] - preds["r"])
    preds["abs_err_g"] = np.abs(preds["pred_g_int"] - preds["g"])
    preds["abs_err_b"] = np.abs(preds["pred_b_int"] - preds["b"])
    preds["mae_rgb_row"] = preds[["abs_err_r", "abs_err_g", "abs_err_b"]].mean(axis=1)

    warning_summary = {
        "out_of_range_row_count": int(bundle.out_of_range_row_count),
        "out_of_range_channel_count": int(bundle.out_of_range_channel_count),
    }
    return preds, warning_summary


def predict_xyz_dataframe(
    model: RGBToXYZMLP,
    df: pd.DataFrame,
    split_name: str,
    *,
    warn: bool,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rgb = df[["r", "g", "b"]].to_numpy(dtype=float)
    bundle = model.predict_xyz(rgb, warn=warn)
    preds = df.copy()
    preds["split"] = split_name
    preds["pred_X"] = bundle.xyz_float[:, 0]
    preds["pred_Y"] = bundle.xyz_float[:, 1]
    preds["pred_Z"] = bundle.xyz_float[:, 2]
    preds["pred_row_invalid_xyz"] = np.any(bundle.invalid_mask, axis=1)
    preds["pred_invalid_xyz_channels"] = np.sum(bundle.invalid_mask, axis=1).astype(int)
    preds["abs_err_X"] = np.abs(preds["pred_X"] - preds["X"])
    preds["abs_err_Y"] = np.abs(preds["pred_Y"] - preds["Y"])
    preds["abs_err_Z"] = np.abs(preds["pred_Z"] - preds["Z"])
    preds["mae_xyz_row"] = preds[["abs_err_X", "abs_err_Y", "abs_err_Z"]].mean(axis=1)

    warning_summary = {
        "invalid_row_count": int(bundle.invalid_row_count),
        "invalid_channel_count": int(bundle.invalid_channel_count),
    }
    return preds, warning_summary


def compute_metrics(preds: pd.DataFrame) -> dict[str, float | int]:
    if preds.empty:
        return {
            "n_rows": 0,
            "mae_rgb_mean": float("nan"),
            "rmse_rgb_mean": float("nan"),
            "mae_r": float("nan"),
            "mae_g": float("nan"),
            "mae_b": float("nan"),
            "rmse_r": float("nan"),
            "rmse_g": float("nan"),
            "rmse_b": float("nan"),
            "out_of_range_row_count": 0,
            "out_of_range_channel_count": 0,
        }

    actual = preds[["r", "g", "b"]].to_numpy(dtype=float)
    predicted = preds[["pred_r_int", "pred_g_int", "pred_b_int"]].to_numpy(dtype=float)
    error = predicted - actual
    mae_channels = np.mean(np.abs(error), axis=0)
    rmse_channels = np.sqrt(np.mean(error**2, axis=0))

    return {
        "n_rows": int(len(preds)),
        "mae_rgb_mean": float(np.mean(np.abs(error))),
        "rmse_rgb_mean": float(np.sqrt(np.mean(error**2))),
        "mae_r": float(mae_channels[0]),
        "mae_g": float(mae_channels[1]),
        "mae_b": float(mae_channels[2]),
        "rmse_r": float(rmse_channels[0]),
        "rmse_g": float(rmse_channels[1]),
        "rmse_b": float(rmse_channels[2]),
        "out_of_range_row_count": int(preds["pred_row_out_of_range"].sum()),
        "out_of_range_channel_count": int(preds["pred_out_of_range_channels"].sum()),
    }


def compute_xyz_metrics(preds: pd.DataFrame) -> dict[str, float | int]:
    if preds.empty:
        return {
            "n_rows": 0,
            "mae_xyz_mean": float("nan"),
            "rmse_xyz_mean": float("nan"),
            "mae_X": float("nan"),
            "mae_Y": float("nan"),
            "mae_Z": float("nan"),
            "rmse_X": float("nan"),
            "rmse_Y": float("nan"),
            "rmse_Z": float("nan"),
            "invalid_row_count": 0,
            "invalid_channel_count": 0,
        }

    actual = preds[["X", "Y", "Z"]].to_numpy(dtype=float)
    predicted = preds[["pred_X", "pred_Y", "pred_Z"]].to_numpy(dtype=float)
    error = predicted - actual
    mae_channels = np.mean(np.abs(error), axis=0)
    rmse_channels = np.sqrt(np.mean(error**2, axis=0))

    return {
        "n_rows": int(len(preds)),
        "mae_xyz_mean": float(np.mean(np.abs(error))),
        "rmse_xyz_mean": float(np.sqrt(np.mean(error**2))),
        "mae_X": float(mae_channels[0]),
        "mae_Y": float(mae_channels[1]),
        "mae_Z": float(mae_channels[2]),
        "rmse_X": float(rmse_channels[0]),
        "rmse_Y": float(rmse_channels[1]),
        "rmse_Z": float(rmse_channels[2]),
        "invalid_row_count": int(preds["pred_row_invalid_xyz"].sum()),
        "invalid_channel_count": int(preds["pred_invalid_xyz_channels"].sum()),
    }


def summarize_ids(preds: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame(
            columns=[
                "id",
                "n_rows",
                "r_target_mean",
                "g_target_mean",
                "b_target_mean",
                "r_pred_mean",
                "g_pred_mean",
                "b_pred_mean",
                "mae_rgb_mean",
            ]
        )

    summary = (
        preds.groupby("id", as_index=False)
        .agg(
            n_rows=("id", "size"),
            r_target_mean=("r", "mean"),
            g_target_mean=("g", "mean"),
            b_target_mean=("b", "mean"),
            r_pred_mean=("pred_r_float", "mean"),
            g_pred_mean=("pred_g_float", "mean"),
            b_pred_mean=("pred_b_float", "mean"),
            mae_rgb_mean=("mae_rgb_row", "mean"),
        )
        .sort_values("id")
        .reset_index(drop=True)
    )
    return summary


def summarize_xyz_ids(preds: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return pd.DataFrame(
            columns=[
                "id",
                "n_rows",
                "X_target_mean",
                "Y_target_mean",
                "Z_target_mean",
                "X_pred_mean",
                "Y_pred_mean",
                "Z_pred_mean",
                "mae_xyz_mean",
            ]
        )

    summary = (
        preds.groupby("id", as_index=False)
        .agg(
            n_rows=("id", "size"),
            X_target_mean=("X", "mean"),
            Y_target_mean=("Y", "mean"),
            Z_target_mean=("Z", "mean"),
            X_pred_mean=("pred_X", "mean"),
            Y_pred_mean=("pred_Y", "mean"),
            Z_pred_mean=("pred_Z", "mean"),
            mae_xyz_mean=("mae_xyz_row", "mean"),
        )
        .sort_values("id")
        .reset_index(drop=True)
    )
    return summary


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def plot_loss_curves(
    history_df: pd.DataFrame,
    out_path: Path,
    *,
    train_metric_column: str,
    test_metric_column: str,
    metric_title: str,
    metric_ylabel: str,
    show: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    axes[0].plot(history_df["step"], history_df["train_data_loss"], label="train")
    if np.isfinite(history_df["test_data_loss"]).any():
        axes[0].plot(history_df["step"], history_df["test_data_loss"], label="test")
    axes[0].set_title("Scaled RGB MSE")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("MSE")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(history_df["step"], history_df[train_metric_column], label="train")
    if np.isfinite(history_df[test_metric_column]).any():
        axes[1].plot(history_df["step"], history_df[test_metric_column], label="test")
    axes[1].set_title(metric_title)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel(metric_ylabel)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_xyz_scatter(preds: pd.DataFrame, title: str, out_path: Path, *, show: bool = False) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    channels = [("X", "pred_X"), ("Y", "pred_Y"), ("Z", "pred_Z")]

    for ax, (target_col, pred_col) in zip(axes, channels, strict=True):
        target = preds[target_col].to_numpy(dtype=float)
        pred = preds[pred_col].to_numpy(dtype=float)
        if target.size == 0:
            ax.axis("off")
            continue

        lo = float(min(np.min(target), np.min(pred)))
        hi = float(max(np.max(target), np.max(pred)))
        pad = 0.05 * max(hi - lo, 1e-6)
        ax.scatter(target, pred, s=24, alpha=0.75)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="black", alpha=0.75)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{target_col} target vs predicted")
        ax.set_xlabel(f"Measured {target_col}")
        ax.set_ylabel(f"Predicted {target_col}")
        ax.grid(alpha=0.25)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_swatches(preds: pd.DataFrame, title: str, out_path: Path, *, show: bool = False) -> None:
    summary = summarize_ids(preds)
    if summary.empty:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No rows in split", ha="center", va="center", fontsize=12)
        fig.suptitle(title)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return

    n = len(summary)
    ncols = 6
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3.1 * ncols, 2.05 * nrows),
        squeeze=False,
    )

    for k in range(nrows * ncols):
        ax = axes[k // ncols, k % ncols]
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if k >= n:
            ax.axis("off")
            continue

        row = summary.iloc[k]
        actual_rgb = np.clip(
            np.array([row["r_target_mean"], row["g_target_mean"], row["b_target_mean"]], dtype=float) / 255.0,
            0.0,
            1.0,
        )
        pred_rgb = np.clip(
            np.array([row["r_pred_mean"], row["g_pred_mean"], row["b_pred_mean"]], dtype=float) / 255.0,
            0.0,
            1.0,
        )

        patch = np.zeros((90, 180, 3), dtype=float)
        patch[:, :90, :] = actual_rgb[None, None, :]
        patch[:, 90:, :] = pred_rgb[None, None, :]
        patch[:, 89:91, :] = 1.0
        ax.imshow(patch, interpolation="nearest")
        ax.set_title(
            f"ID {int(row['id'])} | MAE {row['mae_rgb_mean']:.1f}",
            fontsize=8.5,
            pad=4,
        )

    fig.suptitle(title, fontsize=14)
    fig.text(
        0.5,
        0.01,
        "Each swatch: LEFT = target RGB, RIGHT = model prediction from measured XYZ",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit a standalone XYZ -> RGB MLP screen model for the new screen."
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        default=[str(DEFAULT_INPUT_FILE)],
        help=(
            "One or more input files. Each may be raw spectra (rep,id,r,g,b,nm,power) "
            "or precomputed XYZ (rep,id,r,g,b,X,Y,Z). "
            f"Default: {DEFAULT_INPUT_FILE}"
        ),
    )
    parser.add_argument(
        "--cmf",
        default="1931",
        help="Color matching function for SPD conversion. Default: 1931",
    )
    parser.add_argument(
        "--cmf-file",
        default=None,
        help="Optional CMF file path overriding the built-in default.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=16,
        help="Hidden layer width for the sigmoid MLP. Default: 16",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.95,
        help="Training fraction for the ID-disjoint split. Default: 0.95",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=2026,
        help="Random seed for the ID-disjoint train/test split. Default: 2026",
    )
    parser.add_argument(
        "--white-id",
        type=int,
        default=1,
        help="White-point ID that must place at least one row in training. Default: 1",
    )
    parser.add_argument(
        "--black-id",
        type=int,
        default=0,
        help="Black-point ID that must place at least one row in training. Default: 0",
    )
    parser.add_argument(
        "--extra-guaranteed-train-ids",
        default="156,157",
        help=(
            "Comma/range list of additional IDs that must place at least one row in "
            "training. Default: 156,157"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20000,
        help="Maximum optimization steps. Default: 20000",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-2,
        help="Adam learning rate. Default: 1e-3",
    )
    parser.add_argument(
        "--l2-weight",
        type=float,
        default=5e-4,
        help="L2 regularization weight. Default: 5e-4",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=5.0,
        help="Gradient clipping norm. Default: 5.0",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Record loss metrics every N steps. Default: 100",
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print optimization progress during fitting.",
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

    cmf_key, cmf_path, cmf_wavelengths, cmf_values = load_cmf(args.cmf, args.cmf_file)

    all_xyz_tables: list[pd.DataFrame] = []
    input_summaries: list[dict[str, Any]] = []
    for raw_path_text in args.input_files:
        raw_path = Path(raw_path_text).resolve()
        table = read_table_auto(raw_path)
        source_file = raw_path.name

        if SPD_COLUMNS.issubset(table.columns):
            xyz_df = convert_spd_df_to_xyz(
                table,
                cmf_wavelengths=cmf_wavelengths,
                cmf_values=cmf_values,
                source_file=source_file,
            )
            input_kind = "spd"
        elif XYZ_COLUMNS.issubset(table.columns):
            xyz_df = normalize_xyz_table(table, source_file=source_file)
            input_kind = "xyz"
        else:
            raise ValueError(
                f"{raw_path} does not match supported schemas. "
                "Need either SPD columns or XYZ columns."
            )

        all_xyz_tables.append(xyz_df)
        input_summaries.append(
            {
                "input_file": str(raw_path),
                "input_kind": input_kind,
                "rows_after_conversion": int(len(xyz_df)),
            }
        )

    combined_xyz = pd.concat(all_xyz_tables, ignore_index=True)
    combined_xyz.to_csv(data_dir / "combined_measured_xyz.csv", index=False)

    clean_xyz, clean_summary = clean_xyz_rows(combined_xyz)
    clean_xyz.to_csv(data_dir / "combined_measured_xyz_clean.csv", index=False)
    if clean_summary["n_rows_clean"] < clean_summary["n_rows_raw"]:
        warnings.warn(
            (
                f"Dropped {clean_summary['n_rows_raw'] - clean_summary['n_rows_clean']} "
                "rows with invalid RGB or XYZ values before fitting."
            ),
            RuntimeWarning,
            stacklevel=2,
        )

    guaranteed_train_ids = [int(args.black_id), int(args.white_id)]
    guaranteed_train_ids.extend(parse_id_spec(args.extra_guaranteed_train_ids))
    deduped_ids: list[int] = []
    seen_ids: set[int] = set()
    for color_id in guaranteed_train_ids:
        if color_id not in seen_ids:
            deduped_ids.append(color_id)
            seen_ids.add(color_id)
    guaranteed_train_ids = deduped_ids

    train_df, test_df, split_summary = split_train_test_ids(
        clean_xyz,
        train_fraction=float(args.train_fraction),
        guaranteed_train_ids=guaranteed_train_ids,
        seed=int(args.random_seed),
    )

    missing_guaranteed = [
        row["id"]
        for row in split_summary["guaranteed_train_id_status"]
        if not row["present"]
    ]
    if missing_guaranteed:
        warnings.warn(
            f"Guaranteed-train IDs not present in data: {missing_guaranteed}",
            RuntimeWarning,
            stacklevel=2,
        )

    id_counts = (
        clean_xyz.groupby("id", as_index=False)
        .agg(total_rows=("id", "size"))
        .merge(train_df.groupby("id", as_index=False).agg(train_rows=("id", "size")), on="id", how="left")
        .merge(test_df.groupby("id", as_index=False).agg(test_rows=("id", "size")), on="id", how="left")
        .fillna(0)
        .sort_values("id")
        .reset_index(drop=True)
    )
    for column in ["train_rows", "test_rows"]:
        id_counts[column] = id_counts[column].astype(int)
    id_counts.to_csv(data_dir / "id_split_counts.csv", index=False)

    xyz_to_rgb_model = XYZToRGBMLP(hidden_dim=int(args.hidden_dim))
    history_df = xyz_to_rgb_model.fit(
        train_df[["X", "Y", "Z"]].to_numpy(dtype=float),
        train_df[["r", "g", "b"]].to_numpy(dtype=float),
        xyz_test=test_df[["X", "Y", "Z"]].to_numpy(dtype=float),
        rgb_test=test_df[["r", "g", "b"]].to_numpy(dtype=float),
        max_steps=int(args.max_steps),
        learning_rate=float(args.learning_rate),
        l2_weight=float(args.l2_weight),
        grad_clip_norm=float(args.grad_clip_norm),
        log_every=int(args.log_every),
        verbose=bool(args.verbose),
    )
    history_df.to_csv(data_dir / "training_history.csv", index=False)

    rgb_to_xyz_model = RGBToXYZMLP(hidden_dim=int(args.hidden_dim))
    rgb_to_xyz_history_df = rgb_to_xyz_model.fit(
        train_df[["r", "g", "b"]].to_numpy(dtype=float),
        train_df[["X", "Y", "Z"]].to_numpy(dtype=float),
        rgb_test=test_df[["r", "g", "b"]].to_numpy(dtype=float),
        xyz_test=test_df[["X", "Y", "Z"]].to_numpy(dtype=float),
        max_steps=int(args.max_steps),
        learning_rate=float(args.learning_rate),
        l2_weight=float(args.l2_weight),
        grad_clip_norm=float(args.grad_clip_norm),
        log_every=int(args.log_every),
        verbose=bool(args.verbose),
    )
    rgb_to_xyz_history_df.to_csv(data_dir / "rgb_to_xyz_training_history.csv", index=False)

    model_path = xyz_to_rgb_model.save_state(data_dir / "screen_xyz_to_rgb_mlp_model.pth")
    inverse_model_path = rgb_to_xyz_model.save_state(data_dir / "screen_rgb_to_xyz_mlp_model.pth")

    train_preds, train_warn = predict_dataframe(xyz_to_rgb_model, train_df, "train", warn=True)
    test_preds, test_warn = predict_dataframe(xyz_to_rgb_model, test_df, "test", warn=True)
    all_preds = pd.concat([train_preds, test_preds], ignore_index=True)

    rgb_to_xyz_train_preds, rgb_to_xyz_train_warn = predict_xyz_dataframe(
        rgb_to_xyz_model,
        train_df,
        "train",
        warn=True,
    )
    rgb_to_xyz_test_preds, rgb_to_xyz_test_warn = predict_xyz_dataframe(
        rgb_to_xyz_model,
        test_df,
        "test",
        warn=True,
    )
    rgb_to_xyz_all_preds = pd.concat(
        [rgb_to_xyz_train_preds, rgb_to_xyz_test_preds],
        ignore_index=True,
    )

    train_preds.to_csv(data_dir / "train_predictions.csv", index=False)
    test_preds.to_csv(data_dir / "test_predictions.csv", index=False)
    all_preds.to_csv(data_dir / "all_predictions.csv", index=False)
    rgb_to_xyz_train_preds.to_csv(data_dir / "rgb_to_xyz_train_predictions.csv", index=False)
    rgb_to_xyz_test_preds.to_csv(data_dir / "rgb_to_xyz_test_predictions.csv", index=False)
    rgb_to_xyz_all_preds.to_csv(data_dir / "rgb_to_xyz_all_predictions.csv", index=False)

    train_id_summary = summarize_ids(train_preds)
    test_id_summary = summarize_ids(test_preds)
    all_id_summary = summarize_ids(all_preds)
    train_id_summary.to_csv(data_dir / "train_id_summary.csv", index=False)
    test_id_summary.to_csv(data_dir / "test_id_summary.csv", index=False)
    all_id_summary.to_csv(data_dir / "all_id_summary.csv", index=False)
    rgb_to_xyz_train_id_summary = summarize_xyz_ids(rgb_to_xyz_train_preds)
    rgb_to_xyz_test_id_summary = summarize_xyz_ids(rgb_to_xyz_test_preds)
    rgb_to_xyz_all_id_summary = summarize_xyz_ids(rgb_to_xyz_all_preds)
    rgb_to_xyz_train_id_summary.to_csv(data_dir / "rgb_to_xyz_train_id_summary.csv", index=False)
    rgb_to_xyz_test_id_summary.to_csv(data_dir / "rgb_to_xyz_test_id_summary.csv", index=False)
    rgb_to_xyz_all_id_summary.to_csv(data_dir / "rgb_to_xyz_all_id_summary.csv", index=False)

    plot_loss_curves(
        history_df,
        plots_dir / "01_loss_curves.png",
        train_metric_column="train_mae_rgb",
        test_metric_column="test_mae_rgb",
        metric_title="RGB MAE (0-255 units)",
        metric_ylabel="Mean absolute error",
        show=bool(args.show_plots),
    )
    plot_swatches(
        train_preds,
        "Training split: target vs predicted RGB",
        plots_dir / "02_train_swatches.png",
        show=bool(args.show_plots),
    )
    plot_swatches(
        test_preds,
        "Test split: target vs predicted RGB",
        plots_dir / "03_test_swatches.png",
        show=bool(args.show_plots),
    )
    plot_loss_curves(
        rgb_to_xyz_history_df,
        plots_dir / "04_rgb_to_xyz_loss_curves.png",
        train_metric_column="train_mae_xyz",
        test_metric_column="test_mae_xyz",
        metric_title="XYZ MAE",
        metric_ylabel="Mean absolute error",
        show=bool(args.show_plots),
    )
    plot_xyz_scatter(
        rgb_to_xyz_train_preds,
        "Training split: measured vs predicted XYZ",
        plots_dir / "05_rgb_to_xyz_train_scatter.png",
        show=bool(args.show_plots),
    )
    plot_xyz_scatter(
        rgb_to_xyz_test_preds,
        "Test split: measured vs predicted XYZ",
        plots_dir / "06_rgb_to_xyz_test_scatter.png",
        show=bool(args.show_plots),
    )

    run_config = {
        "input_files": [str(Path(p).resolve()) for p in args.input_files],
        "default_input_file": str(DEFAULT_INPUT_FILE),
        "cmf": cmf_key,
        "cmf_file": str(cmf_path.resolve()),
        "hidden_dim": int(args.hidden_dim),
        "train_fraction": float(args.train_fraction),
        "random_seed": int(args.random_seed),
        "black_id": int(args.black_id),
        "white_id": int(args.white_id),
        "extra_guaranteed_train_ids": parse_id_spec(args.extra_guaranteed_train_ids),
        "max_steps": int(args.max_steps),
        "learning_rate": float(args.learning_rate),
        "l2_weight": float(args.l2_weight),
        "grad_clip_norm": float(args.grad_clip_norm),
        "log_every": int(args.log_every),
        "output_root": str(output_root),
    }
    save_json(data_dir / "run_config.json", run_config)

    xyz_to_rgb_train_metrics = compute_metrics(train_preds)
    xyz_to_rgb_test_metrics = compute_metrics(test_preds)
    xyz_to_rgb_all_metrics = compute_metrics(all_preds)
    rgb_to_xyz_train_metrics = compute_xyz_metrics(rgb_to_xyz_train_preds)
    rgb_to_xyz_test_metrics = compute_xyz_metrics(rgb_to_xyz_test_preds)
    rgb_to_xyz_all_metrics = compute_xyz_metrics(rgb_to_xyz_all_preds)

    fit_summary = {
        "input_summaries": input_summaries,
        "clean_summary": clean_summary,
        "split_summary": split_summary,
        "train_metrics": xyz_to_rgb_train_metrics,
        "test_metrics": xyz_to_rgb_test_metrics,
        "all_metrics": xyz_to_rgb_all_metrics,
        "train_warning_summary": train_warn,
        "test_warning_summary": test_warn,
        "model_fit_summary": xyz_to_rgb_model.fit_summary,
        "model_path": str(model_path.resolve()),
        "xyz_to_rgb": {
            "train_metrics": xyz_to_rgb_train_metrics,
            "test_metrics": xyz_to_rgb_test_metrics,
            "all_metrics": xyz_to_rgb_all_metrics,
            "train_warning_summary": train_warn,
            "test_warning_summary": test_warn,
            "model_fit_summary": xyz_to_rgb_model.fit_summary,
            "model_path": str(model_path.resolve()),
        },
        "rgb_to_xyz": {
            "train_metrics": rgb_to_xyz_train_metrics,
            "test_metrics": rgb_to_xyz_test_metrics,
            "all_metrics": rgb_to_xyz_all_metrics,
            "train_warning_summary": rgb_to_xyz_train_warn,
            "test_warning_summary": rgb_to_xyz_test_warn,
            "model_fit_summary": rgb_to_xyz_model.fit_summary,
            "model_path": str(inverse_model_path.resolve()),
        },
    }
    save_json(data_dir / "fit_summary.json", fit_summary)

    print("Saved fitted model to:", model_path)
    print("Saved inverse model to:", inverse_model_path)
    print("CMF:", cmf_key, "from", cmf_path)
    print("Rows used for fit:", clean_summary["n_rows_clean"])
    print(
        "Train IDs:",
        split_summary["n_ids_train"],
        "| Test IDs:",
        split_summary["n_ids_test"],
    )
    print("Train rows:", len(train_df), "| Test rows:", len(test_df))
    print(
        "XYZ->RGB Train MAE:",
        f"{xyz_to_rgb_train_metrics['mae_rgb_mean']:.4f}",
        "| Test MAE:",
        f"{xyz_to_rgb_test_metrics['mae_rgb_mean']:.4f}",
    )
    print(
        "XYZ->RGB Train out-of-range channels:",
        train_warn["out_of_range_channel_count"],
        "| Test out-of-range channels:",
        test_warn["out_of_range_channel_count"],
    )
    print(
        "RGB->XYZ Train MAE:",
        f"{rgb_to_xyz_train_metrics['mae_xyz_mean']:.6f}",
        "| Test MAE:",
        f"{rgb_to_xyz_test_metrics['mae_xyz_mean']:.6f}",
    )
    print(
        "RGB->XYZ Train invalid channels:",
        rgb_to_xyz_train_warn["invalid_channel_count"],
        "| Test invalid channels:",
        rgb_to_xyz_test_warn["invalid_channel_count"],
    )

if __name__ == "__main__":
    main()
