"""
allan_analysis_fixed.py
-----------------------
Allan deviation analysis for IMU data.

Run:
    python allan_analysis_fixed.py --input imu_data.csv --out allan_plot.png [--fs 400]
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PI = np.pi


def allan_deviation(data: np.ndarray, fs: float):
    N = len(data)
    dt = 1.0 / fs
    max_n = N // 3

    if max_n < 2:
        raise ValueError("Not enough samples for Allan deviation (need at least 6).")

    n_values = np.unique(np.round(np.logspace(0, np.log10(max_n), 400)).astype(int))

    cum = np.cumsum(np.insert(data, 0, 0.0))
    tau_list, adev_list = [], []

    for n in n_values:
        avg = (cum[n:] - cum[:-n]) / n
        if len(avg) < 2 * n:
            break
        diff = avg[n:] - avg[:-n]
        avar = 0.5 * np.mean(diff ** 2)
        tau_list.append(n * dt)
        adev_list.append(np.sqrt(avar))

    return np.array(tau_list), np.array(adev_list)


def geometric_mean_adev(meas: np.ndarray, fs: float):
    per_axis = []
    tau_ref = None

    for ax in range(3):
        tau, adev = allan_deviation(meas[ax], fs)
        if tau_ref is None:
            tau_ref = tau
        per_axis.append(adev)

    avg = np.exp(np.mean(np.log(per_axis), axis=0))
    return tau_ref, avg, per_axis


def fit_line(tau: np.ndarray, offset: float, slope: float) -> np.ndarray:
    return np.exp(offset + slope * np.log(tau))


def _find_slope_region(tau, adev, target_slope, tol=0.15, min_points=8):
    log_tau = np.log10(tau)
    log_adev = np.log10(adev)
    slopes = np.gradient(log_adev, log_tau)

    mask = np.abs(slopes - target_slope) <= tol

    best = None
    start = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= min_points:
                best = (start, i)
                break
            start = None

    if start is not None and len(mask) - start >= min_points:
        best = (start, len(mask))

    return best


def fit_fixed_slope_auto(tau, adev, slope, tol=0.15, min_points=8):
    region = _find_slope_region(tau, adev, slope, tol=tol, min_points=min_points)
    if region is None:
        return np.nan, None

    i0, i1 = region
    log_tau = np.log(tau[i0:i1])
    log_adev = np.log(adev[i0:i1])

    b = np.mean(log_adev - slope * log_tau)
    return float(b), (tau[i0], tau[i1 - 1])


def extract_noise_params(tau: np.ndarray, adev: np.ndarray):
    """
    Extract:
      - white noise (slope -1/2)
      - random walk (slope +1/2)
      - optional bias instability if a flat region exists
    """
    off_wn, wn_range = fit_fixed_slope_auto(tau, adev, -0.5, tol=0.12, min_points=8)
    sigma_wn = np.exp(off_wn) if not np.isnan(off_wn) else np.nan

    off_rw, rw_range = fit_fixed_slope_auto(tau, adev, +0.5, tol=0.12, min_points=8)
    sigma_rw = np.sqrt(3.0) * np.exp(off_rw) if not np.isnan(off_rw) else np.nan

    log_tau = np.log10(tau)
    log_adev = np.log10(adev)
    slopes = np.gradient(log_adev, log_tau)
    flat_mask = np.abs(slopes) <= 0.08

    b_coeff = np.nan
    tau_bi = np.nan
    bi = np.nan
    scf_b = np.sqrt(2 * np.log(2) / PI)

    if np.any(flat_mask):
        idx_flat = np.where(flat_mask)[0]
        idx_bi = idx_flat[np.argmin(adev[idx_flat])]
        tau_bi = tau[idx_bi]
        bi = adev[idx_bi]
        b_coeff = bi / scf_b

    return dict(
        sigma_wn=sigma_wn,
        off_wn=off_wn,
        wn_range=wn_range,
        sigma_rw=sigma_rw,
        off_rw=off_rw,
        rw_range=rw_range,
        bi=bi,
        tau_bi=tau_bi,
        b_coeff=b_coeff,
        scf_b=scf_b,
    )


def plot_adev_panel(ax_plot, tau, adev_avg, adevs, label, unit, params):
    colors = ["tab:blue", "tab:orange", "tab:green"]
    axis_labels = "xyz"

    for i, adev in enumerate(adevs):
        ax_plot.plot(tau, adev, color=colors[i], alpha=0.30, lw=0.9,
                     label=f"axis {axis_labels[i]}")

    ax_plot.plot(tau, adev_avg, "k--", lw=1.8, label="geometric mean")

    p = params

    if not np.isnan(p["off_wn"]):
        if p["wn_range"] is not None:
            tau_fit = tau[(tau >= p["wn_range"][0]) & (tau <= p["wn_range"][1])]
        else:
            tau_fit = tau
        line_wn = fit_line(tau_fit, p["off_wn"], -0.5)
        ax_plot.plot(tau_fit, line_wn, color="red", lw=2,
                     ls="--", label=r"white noise $N$ (slope −1/2)")
        tau_n = 1.0
        val_n = p["sigma_wn"]
        ax_plot.scatter([tau_n], [val_n], marker="o", s=80, color="red", zorder=7)
        ax_plot.annotate(f"N = {val_n:.3e}",
                         xy=(tau_n, val_n), xytext=(tau_n * 1.3, val_n * 1.3),
                         fontsize=8, color="red",
                         arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

    if not np.isnan(p["off_rw"]):
        tau_fit = tau[(tau >= p["rw_range"][0]) & (tau <= p["rw_range"][1])]
        line_rw = fit_line(tau_fit, p["off_rw"], +0.5)
        ax_plot.plot(tau_fit, line_rw, color="purple", lw=2,
                     ls="--", label=r"random walk $K$ (slope +1/2)")
        tau_k = np.sqrt(3.0)
        val_k = p["sigma_rw"]
        ax_plot.scatter([tau_k], [val_k], marker="D", s=70, color="purple", zorder=7)
        ax_plot.annotate(f"K = {val_k:.3e}",
                         xy=(tau_k, val_k), xytext=(tau_k * 1.4, val_k * 0.7),
                         fontsize=8, color="purple",
                         arrowprops=dict(arrowstyle="->", color="purple", lw=0.8))

    if not np.isnan(p["b_coeff"]):
        bi_line = p["scf_b"] * p["b_coeff"]
        ax_plot.axhline(bi_line, color="teal", lw=1.5, ls="--",
                        label=r"bias instability $B$")
        ax_plot.scatter([p["tau_bi"]], [bi_line], marker="^", s=120,
                        color="teal", zorder=7)
        ax_plot.annotate(f"B = {p['b_coeff']:.3e}",
                         xy=(p["tau_bi"], bi_line),
                         xytext=(p["tau_bi"] * 1.4, bi_line * 1.3),
                         fontsize=8, color="teal",
                         arrowprops=dict(arrowstyle="->", color="teal", lw=0.8))

    ax_plot.set_xscale("log")
    ax_plot.set_yscale("log")
    ax_plot.set_xlabel(r"Integration time $\tau$ [s]")
    ax_plot.set_ylabel(f"Allan deviation [{unit}]")
    ax_plot.set_title(label)
    ax_plot.grid(True, which="both", alpha=0.3)
    ax_plot.legend(fontsize=8, loc="lower left")


def print_report(input_path, fs,
                 accel_mean, gyro_mean,
                 params_a, params_g):
    sep = "=" * 68
    print(sep)
    print("  IMU ALLAN DEVIATION ANALYSIS REPORT")
    print(sep)
    print(f"  Input file :  {input_path}")
    print(f"  Sample rate:  {fs:.4f} Hz")
    print()

    print("  ACCELEROMETER")
    print("  " + "-" * 40)
    for i, ax in enumerate("xyz"):
        print(f"    Mean level {ax}:           {accel_mean[i]:+.6e} m/s²")
    print(f"    White-noise coeff N:   {params_a['sigma_wn']:.6e}")
    print(f"    Random-walk coeff K:   {params_a['sigma_rw']:.6e}")
    if not np.isnan(params_a["b_coeff"]):
        print(f"    Bias instability B:    {params_a['b_coeff']:.6e}  (at τ={params_a['tau_bi']:.3f} s)")
    else:
        print("    Bias instability B:    not detected")
    print()

    print("  GYROSCOPE")
    print("  " + "-" * 40)
    for i, ax in enumerate("xyz"):
        print(f"    Mean offset {ax}:          {gyro_mean[i]:+.6e} rad/s")
    print(f"    White-noise coeff N:   {params_g['sigma_wn']:.6e}")
    print(f"    Random-walk coeff K:   {params_g['sigma_rw']:.6e}")
    if not np.isnan(params_g["b_coeff"]):
        print(f"    Bias instability B:    {params_g['b_coeff']:.6e}  (at τ={params_g['tau_bi']:.3f} s)")
    else:
        print("    Bias instability B:    not detected")
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Allan deviation analysis")
    parser.add_argument("--input", type=str, required=True,
                        help="CSV with columns: [t,] ax, ay, az, gx, gy, gz")
    parser.add_argument("--out", type=str, default="allan_deviation.png",
                        help="Output plot path (default: allan_deviation.png)")
    parser.add_argument("--fs", type=float, default=None,
                        help="Sampling rate [Hz]. Inferred from 't' column if omitted.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required = {"ax", "ay", "az", "gx", "gy", "gz"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: Missing columns: {sorted(missing)}")

    if args.fs is not None:
        fs = args.fs
    elif "t" in df.columns:
        t = df["t"].to_numpy()
        dt = np.median(np.diff(t))
        if dt <= 0:
            sys.exit("ERROR: Non-positive dt inferred from 't' column.")
        fs = 1.0 / dt
    else:
        sys.exit("ERROR: Provide --fs or include a 't' column in the CSV.")

    accel_meas = np.vstack([df["ax"], df["ay"], df["az"]])
    gyro_meas = np.vstack([df["gx"], df["gy"], df["gz"]])

    accel_mean = accel_meas.mean(axis=1)
    gyro_mean = gyro_meas.mean(axis=1)

    tau_a, adev_a_avg, adevs_a = geometric_mean_adev(accel_meas, fs)
    tau_g, adev_g_avg, adevs_g = geometric_mean_adev(gyro_meas, fs)

    params_a = extract_noise_params(tau_a, adev_a_avg)
    params_g = extract_noise_params(tau_g, adev_g_avg)

    print_report(args.input, fs, accel_mean, gyro_mean, params_a, params_g)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Allan Deviation Analysis", fontsize=13, fontweight="bold")

    plot_adev_panel(axes[0], tau_a, adev_a_avg, adevs_a,
                    label="Accelerometer", unit="m/s²", params=params_a)
    plot_adev_panel(axes[1], tau_g, adev_g_avg, adevs_g,
                    label="Gyroscope", unit="rad/s", params=params_g)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {args.out}")


if __name__ == "__main__":
    main()