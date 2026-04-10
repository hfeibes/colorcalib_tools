import pandas as pd
from pathlib import Path
import numpy as np
import json
from scipy.optimize import minimize, minimize_scalar

def summarize_xyz_measurements(files):
    """
    Load one or more CSV files containing columns: id, X, Y, Z.
    Exclude rows where any of X, Y, or Z is missing or equals -1.
    Return:
      - valid: cleaned row-level data
      - summary: per-id count, mean, and sample std for X/Y/Z

    Parameters
    ----------
    files : str | Path | list[str | Path]
        One CSV path or a list of CSV paths.

    Returns
    -------
    valid : pandas.DataFrame
        Cleaned measurement rows.
    summary : pandas.DataFrame
        Per-id summary statistics.
    """
    if isinstance(files, (str, Path)):
        files = [files]

    dfs = []
    required = {"id", "X", "Y", "Z"}

    for f in files:
        f = Path(f)
        df = pd.read_csv(f)

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{f} is missing required columns: {sorted(missing)}")

        df = df.copy()
        df["source_file"] = f.name
        dfs.append(df)

    raw = pd.concat(dfs, ignore_index=True)
    raw = raw[["id", "X", "Y", "Z", "source_file"]]

    for col in ["X", "Y", "Z"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    valid = raw.dropna(subset=["X", "Y", "Z"]).copy()
    valid = valid[(valid["X"] != -1) & (valid["Y"] != -1) & (valid["Z"] != -1)].copy()

    summary = (
        valid.groupby("id")
        .agg(
            n=("id", "size"),
            X_mean=("X", "mean"),
            X_std=("X", "std"),
            Y_mean=("Y", "mean"),
            Y_std=("Y", "std"),
            Z_mean=("Z", "mean"),
            Z_std=("Z", "std"),
        )
        .reset_index()
        .sort_values("id")
    )

    return valid, summary


def xyz_to_cie_luv(
    xyz_values,
    reference_point,
):
    """
    Convert one or more XYZ tristimulus values to CIE L*u*v*.

    Parameters
    ----------
    xyz_values : array-like
        Nx3 values in XYZ order.
    reference_point : array-like
        Reference XYZ (Xn, Yn, Zn), e.g., the adaptation/background gray.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: X, Y, Z, L, u, v
    """
    xyz = np.asarray(xyz_values, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz_values must be an Nx3 array-like of XYZ values.")

    reference = np.asarray(reference_point, dtype=float)

    if reference.shape != (3,):
        raise ValueError("reference_point must be length-3 XYZ values.")

    Xn, Yn, Zn = reference
    if Yn <= 0:
        raise ValueError("Reference Y (Yn) must be > 0.")

    den_ref = Xn + 15 * Yn + 3 * Zn
    if den_ref <= 0:
        raise ValueError("reference_point gives invalid u'v' denominator.")

    u_n = (4 * Xn) / den_ref
    v_n = (9 * Yn) / den_ref

    X = xyz[:, 0]
    Y = xyz[:, 1]
    Z = xyz[:, 2]

    den = X + 15 * Y + 3 * Z
    u_prime = np.where(den > 0, (4 * X) / den, 0.0)
    v_prime = np.where(den > 0, (9 * Y) / den, 0.0)

    epsilon = 216 / 24389  # (6/29)^3
    kappa = 24389 / 27     # (29/3)^3
    yr = Y / Yn
    L = np.where(yr > epsilon, 116 * np.cbrt(yr) - 16, kappa * yr)

    u = 13 * L * (u_prime - u_n)
    v = 13 * L * (v_prime - v_n)

    return pd.DataFrame(
        {
            "X": X,
            "Y": Y,
            "Z": Z,
            "L": L,
            "u": u,
            "v": v,
        }
    )


def cie_luv_to_xyz(luv_values, reference_point):
    """
    Convert one or more CIE L*u*v* values to XYZ.

    Parameters
    ----------
    luv_values : array-like
        Nx3 values in [L, u, v] order.
    reference_point : array-like
        Reference XYZ (Xn, Yn, Zn), e.g., adaptation/background gray.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: L, u, v, X, Y, Z
    """
    luv = np.asarray(luv_values, dtype=float)
    if luv.ndim != 2 or luv.shape[1] != 3:
        raise ValueError("luv_values must be an Nx3 array-like of [L, u, v] values.")

    reference = np.asarray(reference_point, dtype=float)
    if reference.shape != (3,):
        raise ValueError("reference_point must be length-3 XYZ values.")

    Xn, Yn, Zn = reference
    if Yn <= 0:
        raise ValueError("Reference Y (Yn) must be > 0.")

    den_ref = Xn + 15 * Yn + 3 * Zn
    if den_ref <= 0:
        raise ValueError("reference_point gives invalid u'v' denominator.")

    u_n = (4 * Xn) / den_ref
    v_n = (9 * Yn) / den_ref

    L = luv[:, 0]
    u = luv[:, 1]
    v = luv[:, 2]

    epsilon = 216 / 24389  # (6/29)^3
    kappa = 24389 / 27     # (29/3)^3

    Y = np.where(L > 8, Yn * np.power((L + 16) / 116, 3), Yn * L / kappa)

    # At L=0 the chromatic terms are undefined; use reference chromaticity.
    safe = L > 1e-12
    u_prime = np.where(safe, u / (13 * L) + u_n, u_n)
    v_prime = np.where(safe, v / (13 * L) + v_n, v_n)

    # X = 9Yu'/(4v'), Z = Y(12 - 3u' - 20v')/(4v')
    denom = 4 * v_prime
    valid = np.abs(denom) > 1e-12

    X = np.zeros_like(Y)
    Z = np.zeros_like(Y)
    X[valid] = 9 * Y[valid] * u_prime[valid] / denom[valid]
    Z[valid] = Y[valid] * (12 - 3 * u_prime[valid] - 20 * v_prime[valid]) / denom[valid]

    return pd.DataFrame(
        {
            "L": L,
            "u": u,
            "v": v,
            "X": X,
            "Y": Y,
            "Z": Z,
        }
    )


class XYZRGBScreenModel:
    """
    Screen model with per-channel nonlinearity and linear RGB<->XYZ transform.

    Default (legacy) mode:
      rgb_lin = (rgb_code / 255) ** gamma_rgb
      xyz_corr = rgb_lin @ M_rgb2xyz
      xyz = xyz_corr + black_xyz

    Robust mode (preferred when calibration ramps are available):
      rgb_lin[c] = trc_c(rgb_code[c] / 255)
      xyz_corr = rgb_lin @ M_rgb2xyz
      xyz = xyz_corr + black_xyz

    where each trc_c is a monotonic piecewise curve fit from single-channel ramps.
    """

    def __init__(
        self,
        black_xyz,
        white_xyz,
        gamma_rgb=None,
        M_rgb2xyz=None,
        fit_rmse_xyz=None,
        trc_code=None,
        trc_linear=None,
        model_variant="gamma_matrix",
    ):
        self.black_xyz = np.asarray(black_xyz, dtype=float).reshape(3)
        self.white_xyz = np.asarray(white_xyz, dtype=float).reshape(3)

        if gamma_rgb is None:
            gamma_rgb = np.array([2.2, 2.2, 2.2], dtype=float)
        self.gamma_rgb = np.asarray(gamma_rgb, dtype=float).reshape(3)

        if M_rgb2xyz is None:
            M_rgb2xyz = np.eye(3, dtype=float)
        self.M_rgb2xyz = np.asarray(M_rgb2xyz, dtype=float).reshape(3, 3)
        self.M_xyz2rgb_lin = np.linalg.pinv(self.M_rgb2xyz)

        self.fit_rmse_xyz = None if fit_rmse_xyz is None else np.asarray(fit_rmse_xyz, dtype=float).reshape(3)

        self.trc_code = self._normalize_trc_list(trc_code, "trc_code")
        self.trc_linear = self._normalize_trc_list(trc_linear, "trc_linear")
        if (self.trc_code is None) != (self.trc_linear is None):
            raise ValueError("trc_code and trc_linear must either both be provided or both be None.")

        self.model_variant = str(model_variant)
        if self.trc_code is not None:
            self.model_variant = "trc_matrix"

    @staticmethod
    def _normalize_trc_list(values, name):
        if values is None:
            return None
        if len(values) != 3:
            raise ValueError(f"{name} must be a list of length 3.")

        out = []
        for i, arr in enumerate(values):
            a = np.asarray(arr, dtype=float).ravel()
            if a.size < 2:
                raise ValueError(f"{name}[{i}] must have at least 2 points.")
            out.append(a)
        return out

    @staticmethod
    def _as_nx3(values, name):
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            if arr.shape[0] != 3:
                raise ValueError(f"{name} must have length 3 or shape Nx3.")
            return arr[None, :], True
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{name} must have shape Nx3.")
        return arr, False

    @staticmethod
    def _interp_with_extrap(x, xp, fp):
        x = np.asarray(x, dtype=float)
        xp = np.asarray(xp, dtype=float)
        fp = np.asarray(fp, dtype=float)

        if xp.ndim != 1 or fp.ndim != 1 or xp.size != fp.size or xp.size < 2:
            raise ValueError("xp/fp must be 1D arrays with same length >= 2.")

        y = np.interp(x, xp, fp)

        dx0 = xp[1] - xp[0]
        dx1 = xp[-1] - xp[-2]
        s0 = (fp[1] - fp[0]) / dx0 if abs(dx0) > 1e-12 else 0.0
        s1 = (fp[-1] - fp[-2]) / dx1 if abs(dx1) > 1e-12 else 0.0

        below = x < xp[0]
        above = x > xp[-1]
        if np.any(below):
            y[below] = fp[0] + (x[below] - xp[0]) * s0
        if np.any(above):
            y[above] = fp[-1] + (x[above] - xp[-1]) * s1

        return y

    def _set_matrix(self, M_rgb2xyz):
        self.M_rgb2xyz = np.asarray(M_rgb2xyz, dtype=float).reshape(3, 3)
        self.M_xyz2rgb_lin = np.linalg.pinv(self.M_rgb2xyz)

    def _set_trc(self, trc_code, trc_linear):
        self.trc_code = self._normalize_trc_list(trc_code, "trc_code")
        self.trc_linear = self._normalize_trc_list(trc_linear, "trc_linear")
        if (self.trc_code is None) != (self.trc_linear is None):
            raise ValueError("trc_code and trc_linear must either both be provided or both be None.")
        if self.trc_code is not None:
            self.model_variant = "trc_matrix"

    def _eval_trc_forward(self, rgb_norm, clip=True):
        out = np.zeros_like(rgb_norm, dtype=float)
        for c in range(3):
            xp = self.trc_code[c]
            fp = self.trc_linear[c]
            x = np.asarray(rgb_norm[:, c], dtype=float)
            if clip:
                x = np.clip(x, xp[0], xp[-1])
                out[:, c] = np.interp(x, xp, fp)
            else:
                out[:, c] = self._interp_with_extrap(x, xp, fp)
        return out

    def _eval_trc_inverse(self, rgb_lin, clip=True):
        out = np.zeros_like(rgb_lin, dtype=float)
        for c in range(3):
            xp = self.trc_linear[c]
            fp = self.trc_code[c]

            # enforce strictly increasing xp for inversion
            xp_u, idx = np.unique(xp, return_index=True)
            fp_u = fp[idx]
            if xp_u.size < 2:
                raise ValueError("TRC inverse is degenerate; not enough unique linear points.")

            x = np.asarray(rgb_lin[:, c], dtype=float)
            if clip:
                x = np.clip(x, xp_u[0], xp_u[-1])
                out[:, c] = np.interp(x, xp_u, fp_u)
            else:
                out[:, c] = self._interp_with_extrap(x, xp_u, fp_u)
        return out

    def _estimate_effective_gamma(self, trc_code, trc_linear):
        gammas = []
        for c in range(3):
            x = np.asarray(trc_code[c], dtype=float)
            y = np.asarray(trc_linear[c], dtype=float)
            mask = (x > 1e-4) & (x < 0.9999) & (y > 1e-6) & (y < 0.9999)
            if np.count_nonzero(mask) < 3:
                gammas.append(2.2)
                continue
            g_vals = np.log(y[mask]) / np.log(x[mask])
            g = float(np.nanmedian(g_vals))
            if not np.isfinite(g):
                g = 2.2
            gammas.append(float(np.clip(g, 0.5, 6.0)))
        return np.asarray(gammas, dtype=float)

    def _prepare_calibration_df(self, calibration_data):
        if isinstance(calibration_data, (str, Path)):
            df = pd.read_csv(calibration_data)
        else:
            df = calibration_data.copy()

        required = {"r", "g", "b", "X", "Y", "Z"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Calibration data is missing required columns: {sorted(missing)}")

        for c in ["r", "g", "b", "X", "Y", "Z"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["r", "g", "b", "X", "Y", "Z"]).copy()
        if df.empty:
            raise ValueError("No valid calibration rows after numeric coercion and NaN removal.")

        # Average repeated measurements for each RGB code triplet.
        df_agg = (
            df.groupby(["r", "g", "b"], as_index=False)[["X", "Y", "Z"]]
            .mean()
            .sort_values(["r", "g", "b"])
            .reset_index(drop=True)
        )
        return df_agg

    def _fit_gamma_matrix(self, df_agg, gamma_init=(2.2, 2.2, 2.2), gamma_bounds=(0.8, 4.0), max_iter=2000):
        rgb_code = df_agg[["r", "g", "b"]].to_numpy(dtype=float)
        rgb_norm = np.clip(rgb_code / 255.0, 0.0, 1.0)
        xyz = df_agg[["X", "Y", "Z"]].to_numpy(dtype=float)
        xyz_corr = np.clip(xyz - self.black_xyz[None, :], 0.0, None)

        g0 = np.asarray(gamma_init, dtype=float).reshape(3)
        lo, hi = float(gamma_bounds[0]), float(gamma_bounds[1])
        if lo <= 0 or hi <= lo:
            raise ValueError("gamma_bounds must satisfy 0 < lower < upper.")

        log_lo, log_hi = np.log(lo), np.log(hi)
        x0 = np.log(np.clip(g0, lo, hi))
        bounds = [(log_lo, log_hi)] * 3

        def _objective(log_g):
            gamma = np.exp(log_g)
            rgb_lin = np.power(rgb_norm, gamma[None, :])
            M, *_ = np.linalg.lstsq(rgb_lin, xyz_corr, rcond=None)
            pred = rgb_lin @ M
            err = pred - xyz_corr
            return float(np.mean(err * err))

        res = minimize(
            _objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(max_iter)},
        )

        gamma = np.exp(res.x)
        rgb_lin = np.power(rgb_norm, gamma[None, :])
        M_rgb2xyz, *_ = np.linalg.lstsq(rgb_lin, xyz_corr, rcond=None)
        pred = rgb_lin @ M_rgb2xyz
        fit_rmse_xyz = np.sqrt(np.mean((pred - xyz_corr) ** 2, axis=0))

        return {
            "variant": "gamma_matrix",
            "gamma_rgb": gamma,
            "M_rgb2xyz": M_rgb2xyz,
            "fit_rmse_xyz": fit_rmse_xyz,
            "trc_code": None,
            "trc_linear": None,
        }

    def _fit_trc_matrix(self, df_agg):
        rgb_code = df_agg[["r", "g", "b"]].to_numpy(dtype=float)
        xyz = df_agg[["X", "Y", "Z"]].to_numpy(dtype=float)
        xyz_corr = np.clip(xyz - self.black_xyz[None, :], 0.0, None)

        chan = ["r", "g", "b"]
        trc_code = []
        trc_linear = []

        for c, name in enumerate(chan):
            other = [v for v in chan if v != name]
            ramp = df_agg.loc[(df_agg[other[0]] == 0) & (df_agg[other[1]] == 0), [name, "X", "Y", "Z"]].copy()
            ramp = ramp.groupby(name, as_index=False)[["X", "Y", "Z"]].mean().sort_values(name)

            if len(ramp) < 4:
                raise ValueError(f"Not enough single-channel ramp points for {name.upper()} channel.")

            code = np.clip(ramp[name].to_numpy(dtype=float) / 255.0, 0.0, 1.0)
            ramp_xyz_corr = np.clip(ramp[["X", "Y", "Z"]].to_numpy(dtype=float) - self.black_xyz[None, :], 0.0, None)

            if code[0] > 0:
                code = np.r_[0.0, code]
                ramp_xyz_corr = np.vstack([np.zeros((1, 3), dtype=float), ramp_xyz_corr])

            direction = ramp_xyz_corr[-1]
            dnorm2 = float(direction @ direction)
            if dnorm2 <= 1e-14:
                raise ValueError(f"Degenerate single-channel ramp for {name.upper()}.")

            scalar = (ramp_xyz_corr @ direction) / dnorm2
            scalar = np.clip(scalar, 0.0, None)
            scalar = np.maximum.accumulate(scalar)
            scalar = scalar - scalar[0]
            if scalar[-1] <= 1e-12:
                raise ValueError(f"Non-varying TRC for {name.upper()}.")

            scalar = scalar / scalar[-1]
            scalar = np.clip(scalar, 0.0, 1.0)
            code[0] = 0.0
            scalar[0] = 0.0

            if code[-1] < 1.0:
                code = np.r_[code, 1.0]
                scalar = np.r_[scalar, 1.0]
            else:
                code[-1] = 1.0
                scalar[-1] = 1.0

            tmp = pd.DataFrame({"code": code, "lin": scalar}).groupby("code", as_index=False)["lin"].max().sort_values("code")
            code = tmp["code"].to_numpy(dtype=float)
            scalar = np.maximum.accumulate(tmp["lin"].to_numpy(dtype=float))
            scalar = np.clip(scalar, 0.0, 1.0)

            if np.unique(scalar).size < 2:
                raise ValueError(f"Degenerate monotonic TRC for {name.upper()}.")

            trc_code.append(code)
            trc_linear.append(scalar)

        rgb_norm = np.clip(rgb_code / 255.0, 0.0, 1.0)
        rgb_lin = np.zeros_like(rgb_norm, dtype=float)
        for c in range(3):
            rgb_lin[:, c] = np.interp(rgb_norm[:, c], trc_code[c], trc_linear[c])

        M_rgb2xyz, *_ = np.linalg.lstsq(rgb_lin, xyz_corr, rcond=None)
        pred = rgb_lin @ M_rgb2xyz
        fit_rmse_xyz = np.sqrt(np.mean((pred - xyz_corr) ** 2, axis=0))
        gamma_eff = self._estimate_effective_gamma(trc_code, trc_linear)

        return {
            "variant": "trc_matrix",
            "gamma_rgb": gamma_eff,
            "M_rgb2xyz": M_rgb2xyz,
            "fit_rmse_xyz": fit_rmse_xyz,
            "trc_code": trc_code,
            "trc_linear": trc_linear,
        }

    def fit(
        self,
        calibration_data,
        gamma_init=(2.2, 2.2, 2.2),
        gamma_bounds=(0.8, 4.0),
        max_iter=2000,
        mode="auto",
    ):
        """
        Fit model from calibration measurements with columns: r, g, b, X, Y, Z.

        Parameters
        ----------
        calibration_data : pandas.DataFrame | str | Path
            Calibration table or CSV path.
        gamma_init : tuple[float, float, float]
            Initial (gamma_R, gamma_G, gamma_B) used by gamma fallback.
        gamma_bounds : tuple[float, float]
            Shared lower/upper bounds for each gamma (gamma fallback).
        max_iter : int
            Maximum optimizer iterations for gamma fallback.
        mode : str
            One of: "auto", "trc_matrix", "gamma_matrix".
            - auto: try robust trc_matrix first, fallback to gamma_matrix.
            - trc_matrix: require robust fit.
            - gamma_matrix: force legacy gamma fit.
        """
        mode = str(mode).lower()
        if mode not in {"auto", "trc_matrix", "gamma_matrix"}:
            raise ValueError("mode must be one of: 'auto', 'trc_matrix', 'gamma_matrix'.")

        df_agg = self._prepare_calibration_df(calibration_data)

        if mode == "gamma_matrix":
            fit_out = self._fit_gamma_matrix(df_agg, gamma_init=gamma_init, gamma_bounds=gamma_bounds, max_iter=max_iter)
        elif mode == "trc_matrix":
            fit_out = self._fit_trc_matrix(df_agg)
        else:
            try:
                fit_out = self._fit_trc_matrix(df_agg)
            except Exception:
                fit_out = self._fit_gamma_matrix(df_agg, gamma_init=gamma_init, gamma_bounds=gamma_bounds, max_iter=max_iter)

        self.model_variant = fit_out["variant"]
        self.gamma_rgb = np.asarray(fit_out["gamma_rgb"], dtype=float).reshape(3)
        self._set_matrix(fit_out["M_rgb2xyz"])
        self.fit_rmse_xyz = np.asarray(fit_out["fit_rmse_xyz"], dtype=float).reshape(3)
        self._set_trc(fit_out.get("trc_code"), fit_out.get("trc_linear"))

        if self.trc_code is None and self.model_variant != "gamma_matrix":
            self.model_variant = "gamma_matrix"

        return self

    def to_dict(self):
        return {
            "model_type": "XYZRGBScreenModel",
            "model_variant": self.model_variant,
            "black_xyz": self.black_xyz.tolist(),
            "white_xyz": self.white_xyz.tolist(),
            "gamma_rgb": self.gamma_rgb.tolist(),
            "M_rgb2xyz": self.M_rgb2xyz.tolist(),
            "fit_rmse_xyz": None if self.fit_rmse_xyz is None else self.fit_rmse_xyz.tolist(),
            "trc_code": None if self.trc_code is None else [v.tolist() for v in self.trc_code],
            "trc_linear": None if self.trc_linear is None else [v.tolist() for v in self.trc_linear],
        }

    @classmethod
    def from_dict(cls, params):
        required = {"black_xyz", "white_xyz", "M_rgb2xyz"}
        missing = required - set(params.keys())
        if missing:
            raise ValueError(f"Model parameters missing required keys: {sorted(missing)}")

        gamma_rgb = params.get("gamma_rgb", [2.2, 2.2, 2.2])
        return cls(
            black_xyz=params["black_xyz"],
            white_xyz=params["white_xyz"],
            gamma_rgb=gamma_rgb,
            M_rgb2xyz=params["M_rgb2xyz"],
            fit_rmse_xyz=params.get("fit_rmse_xyz"),
            trc_code=params.get("trc_code"),
            trc_linear=params.get("trc_linear"),
            model_variant=params.get("model_variant", "gamma_matrix"),
        )

    def save_json(self, path):
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return out_path

    @classmethod
    def load_json(cls, path):
        in_path = Path(path)
        with open(in_path, "r", encoding="utf-8") as f:
            params = json.load(f)
        return cls.from_dict(params)

    def rgb_to_xyz(self, rgb_values):
        rgb, squeeze = self._as_nx3(rgb_values, "rgb_values")
        rgb_norm = np.clip(rgb / 255.0, 0.0, 1.0)

        if self.trc_code is not None:
            rgb_lin = self._eval_trc_forward(rgb_norm, clip=True)
        else:
            rgb_lin = np.power(rgb_norm, self.gamma_rgb[None, :])

        xyz = rgb_lin @ self.M_rgb2xyz + self.black_xyz[None, :]
        return xyz[0] if squeeze else xyz

    def xyz_to_rgb(self, xyz_values, clip=False, as_int=True):
        xyz, squeeze = self._as_nx3(xyz_values, "xyz_values")
        rgb_lin = (xyz - self.black_xyz[None, :]) @ self.M_xyz2rgb_lin

        if self.trc_code is not None:
            rgb_norm = self._eval_trc_inverse(rgb_lin, clip=clip)
            rgb = 255.0 * rgb_norm
        else:
            if clip:
                rgb_lin = np.clip(rgb_lin, 0.0, 1.0)
            rgb = 255.0 * np.power(np.clip(rgb_lin, 0.0, None), 1.0 / self.gamma_rgb[None, :])

        if clip:
            rgb = np.clip(rgb, 0.0, 255.0)
        if as_int:
            rgb = np.round(rgb).astype(int)
        return rgb[0] if squeeze else rgb

    def xyz_to_cie_luv(self, xyz_values, reference_xyz):
        xyz, squeeze = self._as_nx3(xyz_values, "xyz_values")
        luv_df = xyz_to_cie_luv(xyz, reference_xyz)
        out = luv_df[["L", "u", "v"]].to_numpy(dtype=float)
        return out[0] if squeeze else out

    def cie_luv_to_xyz(self, luv_values, reference_xyz):
        luv, squeeze = self._as_nx3(luv_values, "luv_values")
        xyz_df = cie_luv_to_xyz(luv, reference_xyz)
        out = xyz_df[["X", "Y", "Z"]].to_numpy(dtype=float)
        return out[0] if squeeze else out


def fit_uv_circle(
    uv_values,
    center_weight=1.0,
    max_iter=2000,
    tol=1e-7,
):
    """
    Fit a circle to (u, v) points while keeping the center close to (0, 0).

    The objective is:
      mean((distance(point, center) - radius)^2) + center_weight * ||center||^2
    where radius is the mean distance of points to the current center.

    Parameters
    ----------
    uv_values : array-like
        Nx2 array-like with columns [u, v].
    center_weight : float, default 1.0
        Penalty on center distance from origin. Higher values keep center nearer (0, 0).
    max_iter : int, default 2000
        Maximum optimization iterations.
    tol : float, default 1e-7
        Convergence tolerance on gradient norm.

    Returns
    -------
    dict
        {
            "center_u": float,
            "center_v": float,
            "radius": float,
            "rmse": float,
            "objective": float,
            "iterations": int,
            "converged": bool,
            "point_table": pandas.DataFrame,  # columns: u, v, dist_to_center, radial_residual
        }
    """
    pts = np.asarray(uv_values, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("uv_values must be an Nx2 array-like.")
    if pts.shape[0] == 0:
        raise ValueError("uv_values must contain at least one point.")
    if center_weight < 0:
        raise ValueError("center_weight must be >= 0.")

    c = np.zeros(2, dtype=float)
    eps = 1e-12
    converged = False
    objective = np.nan

    def _objective_and_grad(center):
        delta = center[None, :] - pts
        d = np.sqrt((delta * delta).sum(axis=1) + eps)
        r = d.mean()
        residuals = d - r

        # d(distance_i)/d(center) for each point i
        dd_dc = delta / d[:, None]
        dr_dc = dd_dc.mean(axis=0)

        grad_data = (2.0 / len(pts)) * ((residuals[:, None] * (dd_dc - dr_dc[None, :])).sum(axis=0))
        grad_reg = 2.0 * center_weight * center
        grad = grad_data + grad_reg

        mse = np.mean(residuals * residuals)
        obj = mse + center_weight * float(center @ center)
        return obj, grad, d, r, residuals

    for it in range(1, max_iter + 1):
        objective, grad, d, r, residuals = _objective_and_grad(c)
        gnorm = float(np.linalg.norm(grad))
        if gnorm < tol:
            converged = True
            break

        # Backtracking line search.
        step = 1.0
        accepted = False
        for _ in range(30):
            c_try = c - step * grad
            obj_try, _, _, _, _ = _objective_and_grad(c_try)
            if obj_try < objective:
                c = c_try
                accepted = True
                break
            step *= 0.5

        if not accepted:
            break

    objective, _, d, r, residuals = _objective_and_grad(c)
    point_table = pd.DataFrame(
        {
            "u": pts[:, 0],
            "v": pts[:, 1],
            "dist_to_center": d,
            "radial_residual": residuals,
        }
    )

    return {
        "center_u": float(c[0]),
        "center_v": float(c[1]),
        "radius": float(r),
        "rmse": float(np.sqrt(np.mean(residuals * residuals))),
        "objective": float(objective),
        "iterations": int(it),
        "converged": bool(converged),
        "point_table": point_table,
    }


def ellipse_uv_from_t(ellipse_params, t_values):
    """
    Evaluate points on an ellipse at parameter t.

    Parameters
    ----------
    ellipse_params : dict
        Must contain: center_u, center_v, axis_a, axis_b, angle_rad
    t_values : array-like
        Ellipse parameter values in radians.

    Returns
    -------
    numpy.ndarray
        Nx2 array of [u, v] points.
    """
    t = np.asarray(t_values, dtype=float)
    cu = float(ellipse_params["center_u"])
    cv = float(ellipse_params["center_v"])
    a = float(ellipse_params["axis_a"])
    b = float(ellipse_params["axis_b"])
    phi = float(ellipse_params["angle_rad"])

    c = np.cos(phi)
    s = np.sin(phi)
    ct = np.cos(t)
    st = np.sin(t)

    u = cu + a * ct * c - b * st * s
    v = cv + a * ct * s + b * st * c
    return np.column_stack([u, v])


def fit_uv_ellipse(
    uv_values,
    center_weight=1.0,
    axis_balance_weight=0.0,
    max_iter=4000,
):
    """
    Fit a rotated ellipse to (u, v) points while penalizing center offset from (0, 0).

    Objective:
      mean((rho_i - 1)^2)
      + center_weight * ||center||^2
      + axis_balance_weight * (log(axis_a) - log(axis_b))^2
    where rho_i is normalized ellipse radius of point i after rotation/translation.

    Parameters
    ----------
    uv_values : array-like
        Nx2 array-like with columns [u, v].
    center_weight : float, default 1.0
        Penalty on center distance from origin.
    axis_balance_weight : float, default 0.0
        Circularity penalty. Larger values bias axis_a and axis_b to be closer.
        Uses log-axis difference, so the term is scale-invariant.
    max_iter : int, default 4000
        Maximum optimization iterations.

    Returns
    -------
    dict
        {
            "center_u", "center_v", "axis_a", "axis_b", "angle_rad",
            "rmse", "objective", "iterations", "converged", "point_table"
        }
    """
    pts = np.asarray(uv_values, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("uv_values must be an Nx2 array-like.")
    if pts.shape[0] < 3:
        raise ValueError("Need at least 3 points to fit an ellipse.")
    if center_weight < 0:
        raise ValueError("center_weight must be >= 0.")
    if axis_balance_weight < 0:
        raise ValueError("axis_balance_weight must be >= 0.")

    mu = pts.mean(axis=0)
    centered = pts - mu
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    phi0 = float(np.arctan2(evecs[1, 0], evecs[0, 0]))
    std_u = max(float(np.std(pts[:, 0])), 1e-3)
    std_v = max(float(np.std(pts[:, 1])), 1e-3)
    a0 = max(np.sqrt(max(float(evals[0]), 1e-6)) * np.sqrt(2.0), std_u)
    b0 = max(np.sqrt(max(float(evals[1]), 1e-6)) * np.sqrt(2.0), std_v)
    p0 = np.array([mu[0], mu[1], np.log(a0), np.log(b0), phi0], dtype=float)

    eps = 1e-12

    def _objective(p):
        cu, cv, log_a, log_b, phi = p
        a = np.exp(log_a)
        b = np.exp(log_b)
        c = np.cos(phi)
        s = np.sin(phi)

        du = pts[:, 0] - cu
        dv = pts[:, 1] - cv
        x = c * du + s * dv
        y = -s * du + c * dv

        rho = np.sqrt((x / a) ** 2 + (y / b) ** 2 + eps)
        residuals = rho - 1.0
        mse = np.mean(residuals * residuals)
        center_reg = center_weight * (cu * cu + cv * cv)
        axis_reg = axis_balance_weight * (log_a - log_b) ** 2
        return float(mse + center_reg + axis_reg)

    res = minimize(
        _objective,
        p0,
        method="L-BFGS-B",
        options={"maxiter": int(max_iter)},
    )

    cu, cv, log_a, log_b, phi = res.x
    a = float(np.exp(log_a))
    b = float(np.exp(log_b))
    # Keep axis_a >= axis_b for consistent representation.
    if b > a:
        a, b = b, a
        phi = float(phi + np.pi / 2.0)

    c = np.cos(phi)
    s = np.sin(phi)
    du = pts[:, 0] - cu
    dv = pts[:, 1] - cv
    x = c * du + s * dv
    y = -s * du + c * dv
    rho = np.sqrt((x / a) ** 2 + (y / b) ** 2 + eps)
    residuals = rho - 1.0

    point_table = pd.DataFrame(
        {
            "u": pts[:, 0],
            "v": pts[:, 1],
            "rho": rho,
            "ellipse_residual": residuals,
        }
    )

    return {
        "center_u": float(cu),
        "center_v": float(cv),
        "axis_a": float(a),
        "axis_b": float(b),
        "axis_ratio": float(a / b) if b > 0 else float("inf"),
        "angle_rad": float((phi + np.pi) % (2 * np.pi) - np.pi),
        "rmse": float(np.sqrt(np.mean(residuals * residuals))),
        "objective": float(_objective(np.array([cu, cv, np.log(a), np.log(b), phi]))),
        "iterations": int(getattr(res, "nit", -1)),
        "converged": bool(res.success),
        "point_table": point_table,
    }


def project_uv_to_ellipse(
    uv_values,
    ellipse_params,
    grid_size=2048,
):
    """
    Project points to their nearest points on a fitted ellipse.

    Parameters
    ----------
    uv_values : array-like
        Nx2 array-like of [u, v] points.
    ellipse_params : dict
        Ellipse params returned by `fit_uv_ellipse`.
    grid_size : int, default 2048
        Number of coarse samples for initialization.

    Returns
    -------
    pandas.DataFrame
        Columns: u, v, u_proj, v_proj, t, distance
    """
    pts = np.asarray(uv_values, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("uv_values must be an Nx2 array-like.")
    if grid_size < 64:
        raise ValueError("grid_size must be >= 64.")

    ts = np.linspace(0.0, 2.0 * np.pi, int(grid_size), endpoint=False)
    curve = ellipse_uv_from_t(ellipse_params, ts)
    step = 2.0 * np.pi / float(grid_size)

    rows = []
    for p in pts:
        d2 = np.sum((curve - p[None, :]) ** 2, axis=1)
        i0 = int(np.argmin(d2))
        t0 = float(ts[i0])

        def f(t):
            q = ellipse_uv_from_t(ellipse_params, np.array([t]))[0]
            return float(np.sum((q - p) ** 2))

        res = minimize_scalar(f, bounds=(t0 - 2 * step, t0 + 2 * step), method="bounded")
        t_best = float(res.x % (2.0 * np.pi))
        q_best = ellipse_uv_from_t(ellipse_params, np.array([t_best]))[0]
        dist = float(np.linalg.norm(q_best - p))

        rows.append(
            {
                "u": float(p[0]),
                "v": float(p[1]),
                "u_proj": float(q_best[0]),
                "v_proj": float(q_best[1]),
                "t": t_best,
                "distance": dist,
            }
        )

    return pd.DataFrame(rows)
