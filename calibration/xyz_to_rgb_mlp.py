#!/usr/bin/env python3
"""
Standalone XYZ -> RGB MLP model for screen calibration.

The model is intentionally small:

    z-scored XYZ (3)
        -> Linear(hidden_dim)
        -> sigmoid
        -> Linear(3)
        -> scaled RGB in [-1, 1]

RGB values are mapped back to display units as:

    -1 -> 0
     1 -> 255

The class stores the training normalization statistics and can be saved / loaded
independently of any notebook code.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _require_torch():
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for XYZToRGBMLP. Install/fix torch in this environment."
        ) from exc
    return torch


class _SigmoidMLP:
    def __init__(self, torch_module, hidden_dim: int):
        if int(hidden_dim) < 1:
            raise ValueError("hidden_dim must be >= 1.")

        T = torch_module
        self.torch = T
        self.hidden_dim = int(hidden_dim)
        self.fc1 = T.nn.Linear(3, self.hidden_dim)
        self.fc2 = T.nn.Linear(self.hidden_dim, 3)
        T.nn.init.xavier_uniform_(self.fc1.weight)
        T.nn.init.zeros_(self.fc1.bias)
        T.nn.init.xavier_uniform_(self.fc2.weight)
        T.nn.init.zeros_(self.fc2.bias)

    def parameters(self):
        return list(self.fc1.parameters()) + list(self.fc2.parameters())

    def state_dict(self) -> dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "fc1": self.fc1.state_dict(),
            "fc2": self.fc2.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.fc1.load_state_dict(state["fc1"])
        self.fc2.load_state_dict(state["fc2"])

    def train(self) -> None:
        self.fc1.train()
        self.fc2.train()

    def eval(self) -> None:
        self.fc1.eval()
        self.fc2.eval()

    def __call__(self, x_t):
        T = self.torch
        hidden = T.sigmoid(self.fc1(x_t))
        return self.fc2(hidden)


@dataclass(frozen=True)
class PredictionBundle:
    scaled_rgb: np.ndarray
    rgb_float: np.ndarray
    rgb_int: np.ndarray
    out_of_range_mask: np.ndarray
    out_of_range_row_count: int
    out_of_range_channel_count: int


class XYZToRGBMLP:
    """
    Small MLP mapping XYZ coordinates to display RGB.

    Notes
    -----
    - XYZ inputs are z-scored using training-set mean / std.
    - RGB targets are scaled from [0, 255] to [-1, 1] for training.
    - Output predictions are linearly mapped back to [0, 255]. Values outside
      the legal range produce warnings and are clipped before integer rounding.
    """

    def __init__(self, hidden_dim: int = 16):
        self.hidden_dim = int(hidden_dim)
        self.model = None
        self.xyz_mean = np.zeros(3, dtype=float)
        self.xyz_std = np.ones(3, dtype=float)
        self.fit_summary: dict[str, Any] = {}

    @staticmethod
    def _as_nx3(values: Any, name: str) -> tuple[np.ndarray, bool]:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            if arr.shape[0] != 3:
                raise ValueError(f"{name} must have length 3 or shape Nx3.")
            return arr[None, :], True
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{name} must have shape Nx3.")
        return arr, False

    @staticmethod
    def rgb_to_scaled(rgb_values: Any) -> np.ndarray:
        rgb, _ = XYZToRGBMLP._as_nx3(rgb_values, "rgb_values")
        if np.any(~np.isfinite(rgb)):
            raise ValueError("rgb_values must be finite.")
        if np.any((rgb < 0.0) | (rgb > 255.0)):
            raise ValueError("rgb_values must lie within [0, 255].")
        return (rgb / 127.5) - 1.0

    @staticmethod
    def scaled_to_rgb_float(scaled_rgb: Any) -> np.ndarray:
        scaled, _ = XYZToRGBMLP._as_nx3(scaled_rgb, "scaled_rgb")
        return 127.5 * (scaled + 1.0)

    def _normalize_xyz(self, xyz_values: Any) -> np.ndarray:
        xyz, _ = self._as_nx3(xyz_values, "xyz_values")
        if np.any(~np.isfinite(xyz)):
            raise ValueError("xyz_values must be finite.")
        return (xyz - self.xyz_mean[None, :]) / self.xyz_std[None, :]

    def _ensure_model(self):
        if self.model is None:
            raise RuntimeError("Model is not fitted/loaded.")
        return self.model

    def fit(
        self,
        xyz_train: Any,
        rgb_train: Any,
        xyz_test: Any | None = None,
        rgb_test: Any | None = None,
        *,
        max_steps: int = 50000,
        learning_rate: float = 1e-3,
        l2_weight: float = 5e-4,
        grad_clip_norm: float = 5.0,
        log_every: int = 100,
        verbose: bool = False,
    ):
        train_xyz, _ = self._as_nx3(xyz_train, "xyz_train")
        train_rgb, _ = self._as_nx3(rgb_train, "rgb_train")
        if train_xyz.shape[0] != train_rgb.shape[0]:
            raise ValueError("xyz_train and rgb_train must have the same number of rows.")
        if train_xyz.shape[0] < 2:
            raise ValueError("Need at least two training rows.")
        if np.any(~np.isfinite(train_xyz)) or np.any(~np.isfinite(train_rgb)):
            raise ValueError("Training inputs must be finite.")
        if np.any((train_rgb < 0.0) | (train_rgb > 255.0)):
            raise ValueError("Training RGB values must lie within [0, 255].")

        if (xyz_test is None) != (rgb_test is None):
            raise ValueError("xyz_test and rgb_test must either both be provided or both be omitted.")

        if xyz_test is not None:
            test_xyz, _ = self._as_nx3(xyz_test, "xyz_test")
            test_rgb, _ = self._as_nx3(rgb_test, "rgb_test")
            if test_xyz.shape[0] != test_rgb.shape[0]:
                raise ValueError("xyz_test and rgb_test must have the same number of rows.")
            if np.any(~np.isfinite(test_xyz)) or np.any(~np.isfinite(test_rgb)):
                raise ValueError("Test inputs must be finite.")
            if np.any((test_rgb < 0.0) | (test_rgb > 255.0)):
                raise ValueError("Test RGB values must lie within [0, 255].")
        else:
            test_xyz = None
            test_rgb = None

        self.xyz_mean = np.mean(train_xyz, axis=0)
        self.xyz_std = np.std(train_xyz, axis=0)
        self.xyz_std = np.where(self.xyz_std > 1e-8, self.xyz_std, 1.0)

        train_xyz_norm = self._normalize_xyz(train_xyz)
        train_rgb_scaled = self.rgb_to_scaled(train_rgb)

        if test_xyz is not None:
            test_xyz_norm = self._normalize_xyz(test_xyz)
            test_rgb_scaled = self.rgb_to_scaled(test_rgb)
        else:
            test_xyz_norm = None
            test_rgb_scaled = None

        torch = _require_torch()
        self.model = _SigmoidMLP(torch_module=torch, hidden_dim=self.hidden_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(learning_rate))

        x_train_t = torch.tensor(train_xyz_norm, dtype=torch.float32)
        y_train_t = torch.tensor(train_rgb_scaled, dtype=torch.float32)
        if test_xyz_norm is not None:
            x_test_t = torch.tensor(test_xyz_norm, dtype=torch.float32)
            y_test_t = torch.tensor(test_rgb_scaled, dtype=torch.float32)
        else:
            x_test_t = None
            y_test_t = None

        history_rows: list[dict[str, float]] = []
        best_score = None
        best_state = None
        best_step = None
        n_steps = int(max(1, max_steps))
        log_every = max(1, int(log_every))

        for step in range(1, n_steps + 1):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            pred_train = self.model(x_train_t)
            pred_train = torch.nan_to_num(pred_train, nan=0.0, posinf=2.0, neginf=-2.0)

            train_data_loss = torch.mean((pred_train - y_train_t) ** 2)
            reg = 0.0
            for param in self.model.parameters():
                reg = reg + torch.mean(param * param)
            loss = train_data_loss + float(l2_weight) * reg

            if not torch.isfinite(loss):
                raise RuntimeError(f"Encountered non-finite loss at step {step}.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(grad_clip_norm))
            optimizer.step()

            self.model.eval()
            with torch.inference_mode():
                pred_train_eval = self.model(x_train_t)
                pred_train_eval = torch.nan_to_num(pred_train_eval, nan=0.0, posinf=2.0, neginf=-2.0)
                train_eval_loss = torch.mean((pred_train_eval - y_train_t) ** 2)

                train_rgb_eval = 127.5 * (pred_train_eval.detach().cpu().numpy().astype(float) + 1.0)
                train_rgb_eval = np.clip(np.nan_to_num(train_rgb_eval, nan=0.0, posinf=255.0, neginf=0.0), 0.0, 255.0)
                train_mae_rgb = float(np.mean(np.abs(train_rgb_eval - train_rgb)))
                train_rmse_rgb = float(np.sqrt(np.mean((train_rgb_eval - train_rgb) ** 2)))

                if x_test_t is not None:
                    pred_test_eval = self.model(x_test_t)
                    pred_test_eval = torch.nan_to_num(pred_test_eval, nan=0.0, posinf=2.0, neginf=-2.0)
                    test_eval_loss = torch.mean((pred_test_eval - y_test_t) ** 2)

                    test_rgb_eval = 127.5 * (pred_test_eval.detach().cpu().numpy().astype(float) + 1.0)
                    test_rgb_eval = np.clip(np.nan_to_num(test_rgb_eval, nan=0.0, posinf=255.0, neginf=0.0), 0.0, 255.0)
                    test_mae_rgb = float(np.mean(np.abs(test_rgb_eval - test_rgb)))
                    test_rmse_rgb = float(np.sqrt(np.mean((test_rgb_eval - test_rgb) ** 2)))
                    score = float(test_eval_loss.detach().cpu())
                else:
                    test_eval_loss = None
                    test_mae_rgb = float("nan")
                    test_rmse_rgb = float("nan")
                    score = float(train_eval_loss.detach().cpu())

            if best_score is None or score < best_score:
                best_score = score
                best_state = self.model.state_dict()
                best_step = step

            if step == 1 or step % log_every == 0 or step == n_steps:
                row = {
                    "step": float(step),
                    "train_total_loss": float(loss.detach().cpu()),
                    "train_data_loss": float(train_eval_loss.detach().cpu()),
                    "train_mae_rgb": train_mae_rgb,
                    "train_rmse_rgb": train_rmse_rgb,
                    "test_data_loss": float(test_eval_loss.detach().cpu()) if test_eval_loss is not None else float("nan"),
                    "test_mae_rgb": test_mae_rgb,
                    "test_rmse_rgb": test_rmse_rgb,
                }
                history_rows.append(row)
                if verbose:
                    if test_eval_loss is None:
                        print(
                            f"[fit] step={step}/{n_steps} "
                            f"train_loss={row['train_data_loss']:.8f} "
                            f"train_mae_rgb={train_mae_rgb:.4f}"
                        )
                    else:
                        print(
                            f"[fit] step={step}/{n_steps} "
                            f"train_loss={row['train_data_loss']:.8f} "
                            f"test_loss={row['test_data_loss']:.8f} "
                            f"train_mae_rgb={train_mae_rgb:.4f} "
                            f"test_mae_rgb={test_mae_rgb:.4f}"
                        )

        if best_state is None or best_step is None:
            raise RuntimeError("Fit did not produce a valid model state.")

        self.model.load_state_dict(best_state)
        self.model.eval()

        self.fit_summary = {
            "hidden_dim": self.hidden_dim,
            "max_steps": n_steps,
            "learning_rate": float(learning_rate),
            "l2_weight": float(l2_weight),
            "grad_clip_norm": float(grad_clip_norm),
            "best_step": int(best_step),
            "best_score": float(best_score),
            "n_train": int(train_xyz.shape[0]),
            "n_test": int(0 if test_xyz is None else test_xyz.shape[0]),
        }

        import pandas as pd  # noqa: PLC0415

        return pd.DataFrame(history_rows)

    def predict_scaled(self, xyz_values: Any) -> np.ndarray:
        xyz, squeeze = self._as_nx3(xyz_values, "xyz_values")
        x_norm = self._normalize_xyz(xyz)
        model = self._ensure_model()
        torch = _require_torch()
        model.eval()
        with torch.inference_mode():
            x_t = torch.tensor(x_norm, dtype=torch.float32)
            pred = model(x_t).detach().cpu().numpy().astype(float)
        pred = np.nan_to_num(pred, nan=0.0, posinf=2.0, neginf=-2.0)
        return pred[0] if squeeze else pred

    def predict_rgb(
        self,
        xyz_values: Any,
        *,
        warn: bool = True,
        clip: bool = True,
    ) -> PredictionBundle:
        xyz, squeeze = self._as_nx3(xyz_values, "xyz_values")
        scaled = self.predict_scaled(xyz)
        scaled_matrix = np.asarray(scaled, dtype=float).reshape(1, 3) if squeeze else scaled

        rgb_float = self.scaled_to_rgb_float(scaled_matrix)
        out_of_range_mask = (~np.isfinite(rgb_float)) | (rgb_float < 0.0) | (rgb_float > 255.0)
        out_of_range_row_count = int(np.sum(np.any(out_of_range_mask, axis=1)))
        out_of_range_channel_count = int(np.sum(out_of_range_mask))

        if warn and out_of_range_channel_count > 0:
            warnings.warn(
                (
                    f"XYZToRGBMLP predicted {out_of_range_channel_count} channel values "
                    f"outside [0, 255] across {out_of_range_row_count} rows before clipping."
                ),
                RuntimeWarning,
                stacklevel=2,
            )

        rgb_float_safe = np.nan_to_num(rgb_float, nan=0.0, posinf=255.0, neginf=0.0)
        if clip:
            rgb_float_safe = np.clip(rgb_float_safe, 0.0, 255.0)
        rgb_int = np.rint(rgb_float_safe).astype(int)
        rgb_int = np.clip(rgb_int, 0, 255)

        if squeeze:
            return PredictionBundle(
                scaled_rgb=scaled_matrix[0],
                rgb_float=rgb_float_safe[0],
                rgb_int=rgb_int[0],
                out_of_range_mask=out_of_range_mask[0],
                out_of_range_row_count=out_of_range_row_count,
                out_of_range_channel_count=out_of_range_channel_count,
            )

        return PredictionBundle(
            scaled_rgb=scaled_matrix,
            rgb_float=rgb_float_safe,
            rgb_int=rgb_int,
            out_of_range_mask=out_of_range_mask,
            out_of_range_row_count=out_of_range_row_count,
            out_of_range_channel_count=out_of_range_channel_count,
        )

    def state_dict(self) -> dict[str, Any]:
        model = self._ensure_model()
        return {
            "model_type": "XYZToRGBMLP",
            "state_format": "XYZToRGBMLP.sigmoid_hidden.v1",
            "hidden_dim": self.hidden_dim,
            "xyz_mean": self.xyz_mean.tolist(),
            "xyz_std": self.xyz_std.tolist(),
            "fit_summary": self.fit_summary,
            "model_state": model.state_dict(),
        }

    def save_state(self, path: str | Path) -> Path:
        torch = _require_torch()
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out_path)
        return out_path

    @classmethod
    def load_state(cls, path: str | Path) -> "XYZToRGBMLP":
        torch = _require_torch()
        state = torch.load(Path(path), map_location="cpu")
        if state.get("model_type") != "XYZToRGBMLP":
            raise ValueError(f"Unsupported model_type: {state.get('model_type')}")

        model = cls(hidden_dim=int(state["hidden_dim"]))
        model.xyz_mean = np.asarray(state["xyz_mean"], dtype=float)
        model.xyz_std = np.asarray(state["xyz_std"], dtype=float)
        model.xyz_std = np.where(model.xyz_std > 1e-8, model.xyz_std, 1.0)
        model.fit_summary = dict(state.get("fit_summary", {}))

        model.model = _SigmoidMLP(torch_module=torch, hidden_dim=model.hidden_dim)
        model.model.load_state_dict(state["model_state"])
        model.model.eval()
        return model
