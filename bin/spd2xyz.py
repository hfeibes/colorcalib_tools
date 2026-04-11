"""
Convert PR spectra measurements to XYZ values.

This module provides a callable function:
    convert_spd_csv_to_xyz(input_file, output_file, cmf="ciejudd", cmf_file=None)
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


DEFAULT_CIEJUDD_CMF = Path(__file__).resolve().parents[1] / "color_matching_functions" / "ciexyzj.txt"


def _interp_to_1nm_domain(x, y):
    """
    Interpolate y(x) to a 1 nm grid when needed.

    Parameters
    ----------
    x : numpy.ndarray, shape (N,)
    y : numpy.ndarray, shape (N,) or (N, K)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1:
        raise ValueError("Wavelength array must be 1D.")
    if len(x) < 2:
        raise ValueError("Need at least two wavelength samples for interpolation.")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if np.max(np.diff(x)) <= 1.0:
        return x, y

    xi = np.arange(int(np.ceil(x[0])), int(np.floor(x[-1])) + 1, 1, dtype=float)
    spline = CubicSpline(x, y, axis=0)
    yi = spline(xi)
    return xi, yi


def common_domain(x1, y1, x2, y2):
    """
    Put two signals on a shared 1 nm wavelength domain.
    """
    x1i, y1i = _interp_to_1nm_domain(x1, y1)
    x2i, y2i = _interp_to_1nm_domain(x2, y2)

    start = max(x1i[0], x2i[0])
    end = min(x1i[-1], x2i[-1])
    if end < start:
        raise ValueError("No overlapping wavelength range between CMF and spectra.")

    xc = np.arange(int(np.ceil(start)), int(np.floor(end)) + 1, 1, dtype=float)
    y1c = y1i[np.isin(x1i, xc), :]
    y2c = y2i[np.isin(x2i, xc)]
    return xc, y1c, y2c


def _load_cmf(cmf="ciejudd", cmf_file=None):
    cmf_key = str(cmf).strip().lower()
    if cmf_key != "ciejudd":
        raise ValueError(f"Unsupported cmf '{cmf}'. Only 'ciejudd' is currently supported.")

    cmf_path = Path(cmf_file) if cmf_file else DEFAULT_CIEJUDD_CMF
    cmf_df = pd.read_csv(
        cmf_path,
        header=None,
        names=["wavelength", "x_bar", "y_bar", "z_bar"],
    )

    x = cmf_df["wavelength"].to_numpy(dtype=float)
    y = cmf_df[["x_bar", "y_bar", "z_bar"]].to_numpy(dtype=float)
    return x, y


def convert_spd_csv_to_xyz(input_file, output_file, cmf="ciejudd", cmf_file=None):
    """
    Convert spectra CSV to XYZ CSV.

    Parameters
    ----------
    input_file : str | Path
        Path to spectra CSV with columns: rep, id, r, g, b, nm, power
    output_file : str | Path
        Path to output XYZ CSV.
    cmf : str, default "ciejudd"
        Color matching function identifier.
    cmf_file : str | Path | None
        Optional custom CMF file path.

    Returns
    -------
    pandas.DataFrame
        Output table with columns: rep, id, r, g, b, X, Y, Z
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    x1, y1 = _load_cmf(cmf=cmf, cmf_file=cmf_file)
    photodata = pd.read_csv(input_path)

    required_cols = {"rep", "id", "r", "g", "b", "nm", "power"}
    missing_cols = required_cols - set(photodata.columns)
    if missing_cols:
        raise ValueError(f"Input file is missing required columns: {sorted(missing_cols)}")

    all_xyz = []
    for (rep, color_id), color_data in photodata.groupby(["rep", "id"], sort=False):
        color_data = color_data.sort_values("nm")

        r = float(color_data["r"].iloc[0])
        g = float(color_data["g"].iloc[0])
        b = float(color_data["b"].iloc[0])

        if r < 0.0 or g < 0.0 or b < 0.0:
            all_xyz.append([rep, color_id, r, g, b, -1.0, -1.0, -1.0])
            continue

        wavelength_spectra = color_data["nm"].to_numpy(dtype=float)
        spectra_measured = color_data["power"].to_numpy(dtype=float)

        _, xyz_common, spectra_common = common_domain(x1, y1, wavelength_spectra, spectra_measured)
        XYZ = spectra_common @ xyz_common
        all_xyz.append([rep, color_id, r, g, b, float(XYZ[0]), float(XYZ[1]), float(XYZ[2])])

    measured_xyz = pd.DataFrame(all_xyz, columns=["rep", "id", "r", "g", "b", "X", "Y", "Z"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    measured_xyz.to_csv(output_path, index=False)
    return measured_xyz


def main():
    parser = argparse.ArgumentParser(description="Convert PR spectra CSV to XYZ CSV.")
    parser.add_argument("input_file", help="Input spectra CSV path.")
    parser.add_argument("output_file", help="Output XYZ CSV path.")
    parser.add_argument(
        "--cmf",
        default="judd",
        help="Color matching function to use (currently only: judd).",
    )
    parser.add_argument(
        "--cmf-file",
        default=None,
        help="Optional custom CMF file path (overrides built-in default).",
    )
    args = parser.parse_args()

    convert_spd_csv_to_xyz(
        input_file=args.input_file,
        output_file=args.output_file,
        cmf=args.cmf,
        cmf_file=args.cmf_file,
    )


if __name__ == "__main__":
    main()
