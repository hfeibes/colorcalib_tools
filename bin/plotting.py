import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import ConvexHull


def lstar_from_relative_luminance(y_rel):
    """Convert relative luminance Y/Yn to CIE L*."""
    y = np.clip(np.asarray(y_rel, dtype=float), 0.0, None)
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    return np.where(y > epsilon, 116.0 * np.cbrt(y) - 16.0, kappa * y)


def select_level_indices(level_values, mode="low_mid_high"):
    """Pick representative level indices from a numeric list/array."""
    level_ids = np.sort(np.unique(np.asarray(level_values, dtype=int)))
    if level_ids.size == 0:
        raise ValueError("level_values is empty.")

    if mode == "all":
        return level_ids.tolist()
    if mode != "low_mid_high":
        raise ValueError("mode must be 'low_mid_high' or 'all'.")

    low = int(level_ids[0])
    mid = int(level_ids[len(level_ids) // 2])
    high = int(level_ids[-1])

    ordered = []
    for x in [low, mid, high]:
        if x not in ordered:
            ordered.append(x)
    return ordered


def _rgb_strings(rgb_values, clip=True):
    arr = np.asarray(rgb_values, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("rgb_values must be Nx3.")

    if clip:
        arr = np.clip(arr, 0, 255)

    out = []
    for r, g, b in np.rint(arr).astype(int):
        out.append(f"rgb({r},{g},{b})")
    return out


def _mesh_box_from_ranges(x_range, y_range, z_range):
    x0, x1 = float(x_range[0]), float(x_range[1])
    y0, y1 = float(y_range[0]), float(y_range[1])
    z0, z1 = float(z_range[0]), float(z_range[1])

    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )

    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]

    return vertices, i, j, k


def _tight_range(values, frac=0.03, min_pad=0.2):
    a = np.asarray(values, dtype=float)
    lo = float(np.nanmin(a))
    hi = float(np.nanmax(a))
    span = max(hi - lo, 1e-9)
    pad = max(frac * span, min_pad)
    return [lo - pad, hi + pad]


def plot_xyz_gamut_with_levels(
    xyz_df,
    rgb_df,
    level_df,
    rgb_to_xyz_fn,
    grey_xyz=None,
    white_xyz=None,
    black_xyz=None,
    level_indices=None,
    title="Screen gamut in XYZ with selected luminance levels",
    axis_titles=("X", "Y", "Z"),
    marker_size=5,
    gamut_opacity=0.15,
    tight_view=False,
    tight_pad_frac=0.03,
    tight_min_pad=0.2,
    camera_eye=(0.95, 0.95, 0.75),
):
    """
    Plot 3D XYZ points at selected luminance levels with a translucent gamut hull.

    Parameters
    ----------
    xyz_df : DataFrame with columns id, x, y, z
    rgb_df : DataFrame with columns id, r, g, b
    level_df : DataFrame with columns id, l_idx, l_level, hue_id
    rgb_to_xyz_fn : callable that maps Nx3 RGB -> Nx3 XYZ
    grey_xyz : array-like | None
        Optional XYZ for gray reference point.
    white_xyz : array-like | None
        Optional XYZ for white reference point.
    black_xyz : array-like | None
        Optional XYZ for black reference point.
    level_indices : iterable[int] | None
        If None, chooses low/mid/high from l_idx.
    """
    req_xyz = {"id", "x", "y", "z"}
    req_rgb = {"id", "r", "g", "b"}
    req_lvl = {"id", "l_idx", "l_level", "hue_id"}
    if not req_xyz.issubset(xyz_df.columns):
        raise ValueError(f"xyz_df missing columns: {sorted(req_xyz - set(xyz_df.columns))}")
    if not req_rgb.issubset(rgb_df.columns):
        raise ValueError(f"rgb_df missing columns: {sorted(req_rgb - set(rgb_df.columns))}")
    if not req_lvl.issubset(level_df.columns):
        raise ValueError(f"level_df missing columns: {sorted(req_lvl - set(level_df.columns))}")

    plot_df = (
        xyz_df[["id", "x", "y", "z"]]
        .merge(rgb_df[["id", "r", "g", "b"]], on="id", how="inner")
        .merge(level_df[["id", "l_idx", "l_level", "hue_id"]], on="id", how="inner")
    )

    if level_indices is None:
        level_indices = select_level_indices(plot_df["l_idx"].to_numpy(), mode="low_mid_high")
    level_indices = [int(v) for v in level_indices]

    rgb_corners = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 255, 255],
        ],
        dtype=float,
    )
    xyz_corners = np.asarray(rgb_to_xyz_fn(rgb_corners), dtype=float)
    hull = ConvexHull(xyz_corners)

    labels = {
        level_indices[0]: f"Lowest L (idx={level_indices[0]})",
        level_indices[len(level_indices) // 2]: f"Middle L (idx={level_indices[len(level_indices) // 2]})",
        level_indices[-1]: f"Highest L (idx={level_indices[-1]})",
    }
    symbols = ["circle", "square", "diamond", "cross", "x", "triangle-up"]

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=xyz_corners[:, 0],
            y=xyz_corners[:, 1],
            z=xyz_corners[:, 2],
            i=hull.simplices[:, 0],
            j=hull.simplices[:, 1],
            k=hull.simplices[:, 2],
            name="Monitor gamut",
            color="lightblue",
            opacity=float(gamut_opacity),
            flatshading=True,
            hoverinfo="skip",
        )
    )

    ref_points = []
    if grey_xyz is not None:
        ref_points.append(("Grey point", np.asarray(grey_xyz, dtype=float).reshape(3), "gray"))
    if white_xyz is not None:
        ref_points.append(("White point", np.asarray(white_xyz, dtype=float).reshape(3), "white"))
    if black_xyz is not None:
        ref_points.append(("Black point", np.asarray(black_xyz, dtype=float).reshape(3), "black"))

    for name, xyz_ref, color in ref_points:
        fig.add_trace(
            go.Scatter3d(
                x=[xyz_ref[0]],
                y=[xyz_ref[1]],
                z=[xyz_ref[2]],
                mode="markers+text",
                name=name,
                text=[name],
                textposition="top center",
                hovertemplate=(
                    f"{name}<br>"
                    "X=%{x:.4f}<br>"
                    "Y=%{y:.4f}<br>"
                    "Z=%{z:.4f}<extra></extra>"
                ),
                marker=dict(
                    size=7,
                    symbol="x",
                    color=color,
                    line=dict(color="black", width=1.0),
                    opacity=1.0,
                ),
            )
        )

    for idx, li in enumerate(level_indices):
        g = plot_df.loc[plot_df["l_idx"] == li].sort_values("hue_id")
        if g.empty:
            continue

        fig.add_trace(
            go.Scatter3d(
                x=g["x"],
                y=g["y"],
                z=g["z"],
                mode="markers",
                name=labels.get(li, f"Level idx={li}"),
                hovertemplate=(
                    "id=%{customdata[0]}<br>hue_id=%{customdata[1]}<br>"
                    "l_idx=%{customdata[2]}<br>l_level=%{customdata[3]:.5f}<br>"
                    "RGB=(%{customdata[4]}, %{customdata[5]}, %{customdata[6]})<extra></extra>"
                ),
                customdata=g[["id", "hue_id", "l_idx", "l_level", "r", "g", "b"]].to_numpy(),
                marker=dict(
                    size=float(marker_size),
                    symbol=symbols[idx % len(symbols)],
                    color=_rgb_strings(g[["r", "g", "b"]].to_numpy(), clip=True),
                    line=dict(color="black", width=0.5),
                    opacity=0.95,
                ),
            )
        )

    scene = dict(
        xaxis_title=axis_titles[0],
        yaxis_title=axis_titles[1],
        zaxis_title=axis_titles[2],
        aspectmode="data",
        camera=dict(eye=dict(x=float(camera_eye[0]), y=float(camera_eye[1]), z=float(camera_eye[2]))),
    )
    if tight_view and not plot_df.empty:
        view_df = plot_df.loc[plot_df["l_idx"].isin(level_indices)]
        if not view_df.empty:
            x_vals = view_df["x"].to_numpy(dtype=float)
            y_vals = view_df["y"].to_numpy(dtype=float)
            z_vals = view_df["z"].to_numpy(dtype=float)

            for _, xyz_ref, _ in ref_points:
                x_vals = np.r_[x_vals, xyz_ref[0]]
                y_vals = np.r_[y_vals, xyz_ref[1]]
                z_vals = np.r_[z_vals, xyz_ref[2]]

            scene["xaxis"] = dict(range=_tight_range(x_vals, frac=tight_pad_frac, min_pad=tight_min_pad))
            scene["yaxis"] = dict(range=_tight_range(y_vals, frac=tight_pad_frac, min_pad=tight_min_pad))
            scene["zaxis"] = dict(range=_tight_range(z_vals, frac=tight_pad_frac, min_pad=tight_min_pad))

    fig.update_layout(title=title, scene=scene, legend=dict(itemsizing="constant"))
    return fig


def plot_luv_gamut_bounds_with_levels(
    luv_df,
    rgb_df,
    rgb_to_xyz_fn,
    xyz_to_luv_fn,
    reference_xyz,
    level_indices=None,
    title="CIE L*u*v*: selected levels with min/max gamut bounds",
    marker_size=5,
    gamut_opacity=0.12,
    tight_view=True,
    tight_pad_frac=0.03,
    tight_min_pad=0.2,
    camera_eye=(0.95, 0.95, 0.75),
):
    """
    Plot LUV points at selected luminance levels with a min/max LUV bounds box.

    Parameters
    ----------
    luv_df : DataFrame with columns id, l_idx, l_level, hue_id, u, v (optional L)
    rgb_df : DataFrame with columns id, r, g, b
    rgb_to_xyz_fn : callable Nx3 RGB -> Nx3 XYZ
    xyz_to_luv_fn : callable (xyz_values, reference_point) -> DataFrame with L,u,v
    reference_xyz : array-like reference white XYZ
    """
    req_luv = {"id", "l_idx", "l_level", "hue_id", "u", "v"}
    req_rgb = {"id", "r", "g", "b"}
    if not req_luv.issubset(luv_df.columns):
        raise ValueError(f"luv_df missing columns: {sorted(req_luv - set(luv_df.columns))}")
    if not req_rgb.issubset(rgb_df.columns):
        raise ValueError(f"rgb_df missing columns: {sorted(req_rgb - set(rgb_df.columns))}")

    plot_df = luv_df[["id", "l_idx", "l_level", "hue_id", "u", "v"]].merge(
        rgb_df[["id", "r", "g", "b"]], on="id", how="inner"
    )

    if "L" in luv_df.columns:
        plot_df = plot_df.merge(luv_df[["id", "L"]], on="id", how="left")
        if plot_df["L"].isna().any():
            plot_df["L"] = lstar_from_relative_luminance(plot_df["l_level"].to_numpy())
    else:
        plot_df["L"] = lstar_from_relative_luminance(plot_df["l_level"].to_numpy())

    if level_indices is None:
        level_indices = select_level_indices(plot_df["l_idx"].to_numpy(), mode="low_mid_high")
    level_indices = [int(v) for v in level_indices]

    rgb_corners = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 255, 255],
        ],
        dtype=float,
    )
    xyz_corners = np.asarray(rgb_to_xyz_fn(rgb_corners), dtype=float)
    luv_corners = xyz_to_luv_fn(xyz_corners, reference_point=reference_xyz)[["L", "u", "v"]].to_numpy(dtype=float)

    L_range = [float(np.min(luv_corners[:, 0])), float(np.max(luv_corners[:, 0]))]
    u_range = [float(np.min(luv_corners[:, 1])), float(np.max(luv_corners[:, 1]))]
    v_range = [float(np.min(luv_corners[:, 2])), float(np.max(luv_corners[:, 2]))]

    vertices, i, j, k = _mesh_box_from_ranges(L_range, u_range, v_range)

    labels = {
        level_indices[0]: f"Lowest L (idx={level_indices[0]})",
        level_indices[len(level_indices) // 2]: f"Middle L (idx={level_indices[len(level_indices) // 2]})",
        level_indices[-1]: f"Highest L (idx={level_indices[-1]})",
    }
    symbols = ["circle", "square", "diamond", "cross", "x", "triangle-up"]

    fig = go.Figure()
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=i,
            j=j,
            k=k,
            name="LUV gamut bounds (min/max box)",
            color="lightblue",
            opacity=float(gamut_opacity),
            flatshading=True,
            hoverinfo="skip",
        )
    )

    for idx, li in enumerate(level_indices):
        g = plot_df.loc[plot_df["l_idx"] == li].sort_values("hue_id")
        if g.empty:
            continue

        fig.add_trace(
            go.Scatter3d(
                x=g["L"],
                y=g["u"],
                z=g["v"],
                mode="markers",
                name=labels.get(li, f"Level idx={li}"),
                hovertemplate=(
                    "id=%{customdata[0]}<br>hue_id=%{customdata[1]}<br>"
                    "l_idx=%{customdata[2]}<br>l_level=%{customdata[3]:.5f}<br>"
                    "L*=%{x:.3f}<br>u*=%{y:.3f}<br>v*=%{z:.3f}<br>"
                    "RGB=(%{customdata[4]}, %{customdata[5]}, %{customdata[6]})<extra></extra>"
                ),
                customdata=g[["id", "hue_id", "l_idx", "l_level", "r", "g", "b"]].to_numpy(),
                marker=dict(
                    size=float(marker_size),
                    symbol=symbols[idx % len(symbols)],
                    color=_rgb_strings(g[["r", "g", "b"]].to_numpy(), clip=True),
                    line=dict(color="black", width=0.5),
                    opacity=0.95,
                ),
            )
        )

    scene = dict(
        xaxis_title="L*",
        yaxis_title="u*",
        zaxis_title="v*",
        aspectmode="data",
        camera=dict(eye=dict(x=float(camera_eye[0]), y=float(camera_eye[1]), z=float(camera_eye[2]))),
    )
    if tight_view and not plot_df.empty:
        view_df = plot_df.loc[plot_df["l_idx"].isin(level_indices)]
        if not view_df.empty:
            scene["xaxis"] = dict(
                range=_tight_range(view_df["L"].to_numpy(), frac=tight_pad_frac, min_pad=tight_min_pad)
            )
            scene["yaxis"] = dict(
                range=_tight_range(view_df["u"].to_numpy(), frac=tight_pad_frac, min_pad=tight_min_pad)
            )
            scene["zaxis"] = dict(
                range=_tight_range(view_df["v"].to_numpy(), frac=tight_pad_frac, min_pad=tight_min_pad)
            )

    fig.update_layout(title=title, scene=scene, legend=dict(itemsizing="constant"))

    bounds = {
        "L_min": L_range[0],
        "L_max": L_range[1],
        "u_min": u_range[0],
        "u_max": u_range[1],
        "v_min": v_range[0],
        "v_max": v_range[1],
    }
    return fig, bounds
