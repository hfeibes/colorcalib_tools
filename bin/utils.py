import pandas as pd
from pathlib import Path
import numpy as np
import json
from scipy.optimize import minimize, minimize_scalar

XYZ_CMF_DEFAULT = "cie_judd"


def _normalize_xyz_cmf(xyz_cmf):
    """
    Normalize XYZ CMF identifier.

    Notes
    -----
    This project uses CIE Judd-corrected XYZ by default.
    """
    key = str(xyz_cmf).strip().lower().replace("-", "").replace("_", "")
    judd_aliases = {"ciejudd", "judd", "ciejudd1951", "judd1951"}
    if key in judd_aliases:
        return "cie_judd"

    cie1931_aliases = {"cie1931", "1931", "ciexyz1931"}
    if key in cie1931_aliases:
        raise ValueError(
            "CIE 1931 XYZ is not supported in this project. "
            "Use CIE Judd-corrected XYZ."
        )

    raise ValueError(
        f"Unsupported xyz_cmf '{xyz_cmf}'. "
        "Supported identifiers: CIE Judd tags only."
    )

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
    xyz_cmf=XYZ_CMF_DEFAULT,
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
    cmf = _normalize_xyz_cmf(xyz_cmf)
    if cmf != "cie_judd":
        raise ValueError(
            "xyz_to_cie_luv expects CIE Judd-corrected XYZ."
        )

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


def cie_luv_to_xyz(luv_values, reference_point, xyz_cmf=XYZ_CMF_DEFAULT):
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
    cmf = _normalize_xyz_cmf(xyz_cmf)
    if cmf != "cie_judd":
        raise ValueError(
            "cie_luv_to_xyz returns CIE Judd-corrected XYZ."
        )

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


class _SigmoidMLP:
    """Minimal torch MLP: Linear(hidden) -> sigmoid -> Linear(out_dim)."""

    def __init__(self, torch, in_dim, hidden_dim, out_dim):
        self.torch = torch
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        if self.in_dim < 1:
            raise ValueError("in_dim must be >= 1.")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1.")
        if self.out_dim < 1:
            raise ValueError("out_dim must be >= 1.")

        self.fc1_w = torch.nn.Parameter(torch.empty((self.in_dim, self.hidden_dim), dtype=torch.float32), requires_grad=True)
        self.fc1_b = torch.nn.Parameter(torch.zeros((self.hidden_dim,), dtype=torch.float32), requires_grad=True)
        self.fc2_w = torch.nn.Parameter(torch.empty((self.hidden_dim, self.out_dim), dtype=torch.float32), requires_grad=True)
        self.fc2_b = torch.nn.Parameter(torch.zeros((self.out_dim,), dtype=torch.float32), requires_grad=True)

        # Xavier-like init for stable optimization.
        torch.nn.init.xavier_uniform_(self.fc1_w)
        torch.nn.init.xavier_uniform_(self.fc2_w)

    def parameters(self):
        return [self.fc1_w, self.fc1_b, self.fc2_w, self.fc2_b]

    def state_dict(self):
        return {
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "out_dim": self.out_dim,
            "fc1_w": self.fc1_w.detach().cpu(),
            "fc1_b": self.fc1_b.detach().cpu(),
            "fc2_w": self.fc2_w.detach().cpu(),
            "fc2_b": self.fc2_b.detach().cpu(),
        }

    def load_state_dict(self, state):
        T = self.torch
        with T.no_grad():
            self.fc1_w.copy_(state["fc1_w"])
            self.fc1_b.copy_(state["fc1_b"])
            self.fc2_w.copy_(state["fc2_w"])
            self.fc2_b.copy_(state["fc2_b"])

    def __call__(self, x_t):
        T = self.torch
        h = T.sigmoid(x_t @ self.fc1_w + self.fc1_b[None, :])
        y = h @ self.fc2_w + self.fc2_b[None, :]
        return y


class XYZRGBScreenModel:
    """
    Minimal dual-path OLED model with independent fits.

    Architecture:
      xyz_to_rgb path:
        XYZ_rel (3) -> Linear(hidden_dim) -> sigmoid -> Linear(3) -> RGB_norm
      rgb_to_xyz path:
        RGB_norm (3) -> Linear(hidden_dim_forward) -> sigmoid -> Linear(3) -> XYZ_rel

    Notes:
      - Paths are fitted independently (no coupling constraints).
      - white_xyz and black_xyz are preserved as required attributes.
    """

    def __init__(self, black_xyz, white_xyz, xyz_cmf=XYZ_CMF_DEFAULT):
        self.black_xyz = np.asarray(black_xyz, dtype=float).reshape(3)
        self.white_xyz = np.asarray(white_xyz, dtype=float).reshape(3)
        self.xyz_cmf = _normalize_xyz_cmf(xyz_cmf)
        if self.xyz_cmf != "cie_judd":
            raise ValueError(
                "XYZRGBScreenModel currently supports CIE Judd-corrected XYZ only."
            )

        self.inverse_model = None
        self.forward_model = None
        self.hidden_dim = None
        self.hidden_dim_forward = None
        self.inverse_input_mean = np.zeros(3, dtype=float)
        self.inverse_input_std = np.ones(3, dtype=float)
        self.forward_input_mean = np.zeros(3, dtype=float)
        self.forward_input_std = np.ones(3, dtype=float)
        # Backward-compatible aliases for prior notebooks.
        self.input_mean = self.inverse_input_mean.copy()
        self.input_std = self.inverse_input_std.copy()

        # Kept for notebook compatibility.
        self.gamma_rgb = np.full(3, np.nan, dtype=float)
        self.fit_rmse_xyz = np.full(3, np.nan, dtype=float)
        self.inverse_fit_rmse_rgb = None
        self.forward_fit_rmse_xyz = None
        self.model_variant = "unfit"

    @staticmethod
    def _require_torch():
        try:
            import torch  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "PyTorch is required for XYZRGBScreenModel fit/save/load. "
                "Install/fix torch runtime in this environment."
            ) from exc
        return torch

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

    def _xyz_span(self):
        return np.maximum(self.white_xyz - self.black_xyz, 1e-8)

    def _xyz_to_relative(self, xyz):
        xyz = np.asarray(xyz, dtype=float)
        return (xyz - self.black_xyz[None, :]) / self._xyz_span()[None, :]

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
        if "id" in df.columns:
            df["id"] = pd.to_numeric(df["id"], errors="coerce")
        else:
            df["id"] = np.nan

        df = df.dropna(subset=["r", "g", "b", "X", "Y", "Z"]).copy()
        # Drop known sentinel/invalid rows used by measurement scripts.
        df = df[(df["X"] != -1) & (df["Y"] != -1) & (df["Z"] != -1)].copy()
        df = df[(df["r"] >= 0) & (df["g"] >= 0) & (df["b"] >= 0)].copy()
        df = df[(df["r"] <= 255) & (df["g"] <= 255) & (df["b"] <= 255)].copy()
        for c in ["r", "g", "b", "X", "Y", "Z"]:
            df = df[np.isfinite(df[c])].copy()
        if df.empty:
            raise ValueError("No valid calibration rows after numeric coercion and NaN removal.")
        return df.reset_index(drop=True)

    def _build_inverse_model(self, hidden_dim):
        torch = self._require_torch()
        return _SigmoidMLP(torch=torch, in_dim=3, hidden_dim=int(hidden_dim), out_dim=3)

    def _build_forward_model(self, hidden_dim):
        torch = self._require_torch()
        return _SigmoidMLP(torch=torch, in_dim=3, hidden_dim=int(hidden_dim), out_dim=3)

    def _fit_mlp_path(
        self,
        model,
        x_t,
        y_t,
        w_t,
        pm_t,
        privileged_weight,
        nonlinear_reg_strength,
        torch_lr,
        nonlinear_max_iter,
        torch_verbose,
        tag,
    ):
        torch = self._require_torch()
        optim = torch.optim.Adam(model.parameters(), lr=float(torch_lr))

        best_loss = None
        best_state = None
        n_steps = int(max(1, nonlinear_max_iter))

        for step in range(n_steps):
            optim.zero_grad(set_to_none=True)
            pred = model(x_t)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)

            data_term = (w_t * ((pred - y_t) ** 2)).mean()
            if float(privileged_weight) > 0.0:
                l1 = torch.abs(pred - y_t)
                denom = torch.clamp(pm_t.sum(), min=1.0)
                privileged_term = (pm_t * l1).sum() / (denom * y_t.shape[1])
            else:
                privileged_term = 0.0 * data_term

            reg = 0.0
            for p in model.parameters():
                reg = reg + torch.mean(p * p)

            loss = data_term + float(privileged_weight) * privileged_term + float(nonlinear_reg_strength) * reg
            if not torch.isfinite(loss):
                if best_state is not None:
                    model.load_state_dict(best_state)
                    break
                raise RuntimeError(f"Non-finite loss during {tag} fit.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()

            any_bad = False
            for p in model.parameters():
                if not torch.isfinite(p).all():
                    any_bad = True
                    break
            if any_bad:
                if best_state is not None:
                    model.load_state_dict(best_state)
                    break
                raise RuntimeError(f"Non-finite parameter encountered during {tag} fit.")

            lv = float(loss.detach().cpu())
            if best_loss is None or lv < best_loss:
                best_loss = lv
                best_state = model.state_dict()

            if torch_verbose and ((step + 1) % max(1, n_steps // 10) == 0 or step == 0):
                print(f"[torch-fit:{tag}] step={step+1}/{n_steps} loss={lv:.8f}")

        if best_state is None:
            raise RuntimeError(f"{tag} fit failed to produce a valid checkpoint.")
        model.load_state_dict(best_state)
        return model

    def fit(
        self,
        calibration_data,
        fit_inverse_model=True,
        fit_forward_model=False,
        privileged_ids=None,
        privileged_weight=0.0,
        hidden_dim=8,
        hidden_dim_forward=None,
        nonlinear_max_iter=18000,
        nonlinear_reg_strength=1e-4,
        torch_lr=1e-3,
        torch_verbose=False,
        **_unused,
    ):
        if not fit_inverse_model and not fit_forward_model:
            raise ValueError("At least one of fit_inverse_model or fit_forward_model must be True.")

        df = self._prepare_calibration_df(calibration_data)
        xyz = df[["X", "Y", "Z"]].to_numpy(dtype=float)
        xyz_rel = self._xyz_to_relative(xyz)
        rgb_norm = np.clip(df[["r", "g", "b"]].to_numpy(dtype=float) / 255.0, 0.0, 1.0)

        torch = self._require_torch()

        if privileged_ids is None:
            privileged_mask = np.zeros(len(df), dtype=bool)
        else:
            priv = set(np.asarray(list(privileged_ids), dtype=float).tolist())
            privileged_mask = df["id"].isin(priv).to_numpy(dtype=bool)
        sample_w = np.ones(len(df), dtype=float)

        w_t = torch.tensor(sample_w[:, None], dtype=torch.float32)
        pm_t = torch.tensor(privileged_mask[:, None], dtype=torch.float32)

        if fit_inverse_model:
            self.inverse_input_mean = np.mean(xyz_rel, axis=0)
            self.inverse_input_std = np.std(xyz_rel, axis=0)
            self.inverse_input_std = np.where(self.inverse_input_std > 1e-6, self.inverse_input_std, 1.0)
            self.input_mean = self.inverse_input_mean.copy()
            self.input_std = self.inverse_input_std.copy()
            x_inv = (xyz_rel - self.inverse_input_mean[None, :]) / self.inverse_input_std[None, :]
            x_inv_t = torch.tensor(x_inv, dtype=torch.float32)
            y_inv_t = torch.tensor(rgb_norm, dtype=torch.float32)

            self.hidden_dim = int(hidden_dim)
            self.inverse_model = self._build_inverse_model(self.hidden_dim)
            self.inverse_model = self._fit_mlp_path(
                model=self.inverse_model,
                x_t=x_inv_t,
                y_t=y_inv_t,
                w_t=w_t,
                pm_t=pm_t,
                privileged_weight=privileged_weight,
                nonlinear_reg_strength=nonlinear_reg_strength,
                torch_lr=torch_lr,
                nonlinear_max_iter=nonlinear_max_iter,
                torch_verbose=torch_verbose,
                tag="xyz_to_rgb_mlp",
            )

            pred_rgb = self._predict_inverse(xyz)
            self.inverse_fit_rmse_rgb = np.sqrt(np.mean((pred_rgb - rgb_norm) ** 2, axis=0))

        if fit_forward_model:
            if hidden_dim_forward is None:
                hidden_dim_forward = hidden_dim
            self.forward_input_mean = np.mean(rgb_norm, axis=0)
            self.forward_input_std = np.std(rgb_norm, axis=0)
            self.forward_input_std = np.where(self.forward_input_std > 1e-6, self.forward_input_std, 1.0)
            x_fwd = (rgb_norm - self.forward_input_mean[None, :]) / self.forward_input_std[None, :]
            x_fwd_t = torch.tensor(x_fwd, dtype=torch.float32)
            y_fwd_t = torch.tensor(xyz_rel, dtype=torch.float32)

            self.hidden_dim_forward = int(hidden_dim_forward)
            self.forward_model = self._build_forward_model(self.hidden_dim_forward)
            self.forward_model = self._fit_mlp_path(
                model=self.forward_model,
                x_t=x_fwd_t,
                y_t=y_fwd_t,
                w_t=w_t,
                pm_t=pm_t,
                privileged_weight=privileged_weight,
                nonlinear_reg_strength=nonlinear_reg_strength,
                torch_lr=torch_lr,
                nonlinear_max_iter=nonlinear_max_iter,
                torch_verbose=torch_verbose,
                tag="rgb_to_xyz_mlp",
            )

            pred_xyz = self.rgb_to_xyz(df[["r", "g", "b"]].to_numpy(dtype=float), clip=False)
            self.forward_fit_rmse_xyz = np.sqrt(np.mean((pred_xyz - xyz) ** 2, axis=0))

        self.gamma_rgb = np.full(3, np.nan, dtype=float)
        self.fit_rmse_xyz = np.full(3, np.nan, dtype=float)
        if fit_inverse_model and fit_forward_model:
            self.model_variant = "dual_mlp_sigmoid_hidden"
        elif fit_inverse_model:
            self.model_variant = "xyz_to_rgb_mlp_sigmoid_hidden"
        else:
            self.model_variant = "rgb_to_xyz_mlp_sigmoid_hidden"
        return self

    def _predict_inverse(self, xyz):
        if self.inverse_model is None:
            raise RuntimeError(
                "xyz_to_rgb requires a fitted/loaded inverse model. "
                "Call fit(..., fit_inverse_model=True) or load_state(...)."
            )
        torch = self._require_torch()
        x = self._xyz_to_relative(np.asarray(xyz, dtype=float))
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("xyz must have shape Nx3.")
        x = (x - self.inverse_input_mean[None, :]) / self.inverse_input_std[None, :]

        with torch.inference_mode():
            x_t = torch.tensor(x, dtype=torch.float32)
            pred = self.inverse_model(x_t).detach().cpu().numpy().astype(float)

        pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
        return pred

    def _predict_forward_relative(self, rgb):
        if self.forward_model is None:
            raise RuntimeError(
                "rgb_to_xyz requires a fitted/loaded forward model. "
                "Call fit(..., fit_forward_model=True) or load_state(...)."
            )
        torch = self._require_torch()
        rgb_norm = np.asarray(rgb, dtype=float) / 255.0
        if rgb_norm.ndim != 2 or rgb_norm.shape[1] != 3:
            raise ValueError("rgb must have shape Nx3.")
        rgb_norm = np.clip(rgb_norm, 0.0, 1.0)
        x = (rgb_norm - self.forward_input_mean[None, :]) / self.forward_input_std[None, :]

        with torch.inference_mode():
            x_t = torch.tensor(x, dtype=torch.float32)
            pred_rel = self.forward_model(x_t).detach().cpu().numpy().astype(float)

        pred_rel = np.nan_to_num(pred_rel, nan=0.0, posinf=1.0, neginf=0.0)
        return pred_rel

    def state_dict(self):
        return {
            "model_type": "XYZRGBScreenModel",
            "state_format": "XYZRGBScreenModel.dual_mlp.v1",
            "xyz_cmf": self.xyz_cmf,
            "black_xyz": self.black_xyz.tolist(),
            "white_xyz": self.white_xyz.tolist(),
            "hidden_dim": self.hidden_dim,
            "hidden_dim_forward": self.hidden_dim_forward,
            "input_mean": self.inverse_input_mean.tolist(),
            "input_std": self.inverse_input_std.tolist(),
            "inverse_input_mean": self.inverse_input_mean.tolist(),
            "inverse_input_std": self.inverse_input_std.tolist(),
            "forward_input_mean": self.forward_input_mean.tolist(),
            "forward_input_std": self.forward_input_std.tolist(),
            "gamma_rgb": self.gamma_rgb.tolist(),
            "fit_rmse_xyz": self.fit_rmse_xyz.tolist(),
            "inverse_fit_rmse_rgb": None if self.inverse_fit_rmse_rgb is None else self.inverse_fit_rmse_rgb.tolist(),
            "forward_fit_rmse_xyz": None if self.forward_fit_rmse_xyz is None else self.forward_fit_rmse_xyz.tolist(),
            "model_variant": self.model_variant,
            "inverse_model_state": None if self.inverse_model is None else self.inverse_model.state_dict(),
            "forward_model_state": None if self.forward_model is None else self.forward_model.state_dict(),
        }

    def save_state(self, path):
        torch = self._require_torch()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out_path)
        return out_path

    @classmethod
    def load_state(cls, path):
        torch = cls._require_torch()
        state = torch.load(Path(path), map_location="cpu")

        model = cls(
            black_xyz=state["black_xyz"],
            white_xyz=state["white_xyz"],
            xyz_cmf=state.get("xyz_cmf", XYZ_CMF_DEFAULT),
        )

        h_inv = state.get("hidden_dim", None)
        model.hidden_dim = None if h_inv is None else int(h_inv)
        h_fwd = state.get("hidden_dim_forward", None)
        model.hidden_dim_forward = None if h_fwd is None else int(h_fwd)
        model.inverse_input_mean = np.asarray(
            state.get("inverse_input_mean", state.get("input_mean", [0.0, 0.0, 0.0])),
            dtype=float,
        )
        model.inverse_input_std = np.asarray(
            state.get("inverse_input_std", state.get("input_std", [1.0, 1.0, 1.0])),
            dtype=float,
        )
        model.inverse_input_std = np.where(model.inverse_input_std > 1e-6, model.inverse_input_std, 1.0)
        model.forward_input_mean = np.asarray(state.get("forward_input_mean", [0.0, 0.0, 0.0]), dtype=float)
        model.forward_input_std = np.asarray(state.get("forward_input_std", [1.0, 1.0, 1.0]), dtype=float)
        model.forward_input_std = np.where(model.forward_input_std > 1e-6, model.forward_input_std, 1.0)
        model.input_mean = model.inverse_input_mean.copy()
        model.input_std = model.inverse_input_std.copy()
        model.gamma_rgb = np.asarray(state.get("gamma_rgb", [np.nan, np.nan, np.nan]), dtype=float)
        model.fit_rmse_xyz = np.asarray(state.get("fit_rmse_xyz", [np.nan, np.nan, np.nan]), dtype=float)
        inv_rmse = state.get("inverse_fit_rmse_rgb")
        model.inverse_fit_rmse_rgb = None if inv_rmse is None else np.asarray(inv_rmse, dtype=float)
        fwd_rmse = state.get("forward_fit_rmse_xyz")
        model.forward_fit_rmse_xyz = None if fwd_rmse is None else np.asarray(fwd_rmse, dtype=float)
        model.model_variant = state.get("model_variant", "loaded")

        inv_state = state.get("inverse_model_state")
        if inv_state is not None:
            model.inverse_model = model._build_inverse_model(model.hidden_dim)
            model.inverse_model.load_state_dict(inv_state)
        else:
            model.inverse_model = None

        fwd_state = state.get("forward_model_state")
        if fwd_state is not None:
            if model.hidden_dim_forward is None:
                model.hidden_dim_forward = int(fwd_state.get("hidden_dim", model.hidden_dim or 8))
            model.forward_model = model._build_forward_model(model.hidden_dim_forward)
            model.forward_model.load_state_dict(fwd_state)
        else:
            model.forward_model = None

        if model.inverse_model is None and model.forward_model is None:
            raise ValueError("Loaded state does not contain inverse_model_state or forward_model_state.")
        return model

    def rgb_to_xyz(self, rgb_values, clip=False):
        rgb, squeeze = self._as_nx3(rgb_values, "rgb_values")
        xyz_rel = self._predict_forward_relative(rgb)
        xyz = self.black_xyz[None, :] + xyz_rel * self._xyz_span()[None, :]
        if clip:
            xyz = np.clip(xyz, self.black_xyz[None, :], self.white_xyz[None, :])
        return xyz[0] if squeeze else xyz

    def xyz_to_rgb(self, xyz_values, clip=False, as_int=True):
        xyz, squeeze = self._as_nx3(xyz_values, "xyz_values")
        rgb_norm = self._predict_inverse(xyz)
        rgb = 255.0 * rgb_norm

        if clip:
            rgb = np.clip(rgb, 0.0, 255.0)
        if as_int:
            rgb = np.round(rgb).astype(int)
        return rgb[0] if squeeze else rgb

    def xyz_to_cie_luv(self, xyz_values, reference_xyz):
        xyz, squeeze = self._as_nx3(xyz_values, "xyz_values")
        luv_df = xyz_to_cie_luv(xyz, reference_xyz, xyz_cmf=self.xyz_cmf)
        out = luv_df[["L", "u", "v"]].to_numpy(dtype=float)
        return out[0] if squeeze else out

    def cie_luv_to_xyz(self, luv_values, reference_xyz):
        luv, squeeze = self._as_nx3(luv_values, "luv_values")
        xyz_df = cie_luv_to_xyz(luv, reference_xyz, xyz_cmf=self.xyz_cmf)
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
