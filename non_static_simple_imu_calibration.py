"""
non_static_simple_imu_calibration.py
====================================

Full non-static IMU calibration validation pipeline on simple trajectories.

Pipeline
--------
1. Generate a simple 2D trajectory from waypoints.
2. Build clean IMU from physics:
      - accelerometer: body-frame specific force
      - gyroscope: body-frame angular rate
3. Inject known per-axis scale, bias, and white noise:
      y = s*x + b + n
4. Estimate scale and bias back using least squares.
5. Correct the corrupted IMU using the estimated parameters.
6. Reconstruct orientation, velocity, and position from corrected IMU.
7. Compare reconstructed trajectory against ground truth.

Run
---
python non_static_simple_imu_calibration.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from waypoints import gen_wp

GRAVITY = 9.81

# ---------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------
OUTPUT_DATA_CSV = "simulated_nonstatic_full_pipeline.csv"
OUTPUT_REPORT_CSV = "nonstatic_full_calibration_report.csv"

ACCEL_BIAS = np.array([0.05, -0.03, 0.02], dtype=float)
GYRO_BIAS = np.array([0.01, -0.015, 0.008], dtype=float)

ACCEL_SCALE = np.array([1.02, 0.98, 1.01], dtype=float)
GYRO_SCALE = np.array([1.01, 0.99, 1.03], dtype=float)

ACCEL_NOISE_STD = np.array([0.01, 0.01, 0.01], dtype=float)
GYRO_NOISE_STD = np.array([0.003, 0.003, 0.003], dtype=float)

SEED = 42


# ---------------------------------------------------------------------
# TRAJECTORY + CLEAN IMU
# ---------------------------------------------------------------------
def build_time_parameterized_trajectory(
    stp, edp, num_points=400, c_type=3, total_time=40.0
):
    """
    Build a simple 2D trajectory from waypoints with uniform time spacing.
    z is kept at 0.
    """
    xy = gen_wp(stp, edp, num_points=num_points, c_type=c_type)
    z = np.zeros((xy.shape[0], 1))
    pos = np.hstack([xy, z])

    t = np.linspace(0.0, total_time, num_points)
    dt = t[1] - t[0]
    return t, dt, pos


def compute_clean_imu_from_positions(dt, pos):
    """
    From position samples, compute:
      - velocity in world frame
      - acceleration in world frame
      - heading from velocity direction
      - gyroscope z-rate from heading derivative
      - body-frame specific force

    Assumptions:
      - planar motion in x-y
      - body x-axis points along velocity direction
      - roll = pitch = 0
      - gravity acts along world z
    """
    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)

    vx = vel[:, 0]
    vy = vel[:, 1]

    heading = np.unwrap(np.arctan2(vy, vx))
    yaw_rate = np.gradient(heading, dt)

    gyro_clean = np.column_stack([
        np.zeros_like(yaw_rate),
        np.zeros_like(yaw_rate),
        yaw_rate
    ])

    g_world = np.array([0.0, 0.0, -GRAVITY])
    spec_force_world = acc - g_world

    accel_clean = np.zeros_like(spec_force_world)

    for i, psi in enumerate(heading):
        c = np.cos(psi)
        s = np.sin(psi)

        # world -> body rotation for yaw-only
        R_bw = np.array([
            [ c,  s, 0.0],
            [-s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])
        accel_clean[i] = R_bw @ spec_force_world[i]

    return vel, acc, heading, gyro_clean, accel_clean


# ---------------------------------------------------------------------
# INJECTION / ESTIMATION / CORRECTION
# ---------------------------------------------------------------------
def inject_sensor_errors(clean_xyz, scale_xyz, bias_xyz, noise_std_xyz, rng):
    """
    Inject:
        corrupted = scale * clean + bias + white noise
    """
    corrupted = np.zeros_like(clean_xyz)
    for i in range(3):
        noise = rng.normal(0.0, noise_std_xyz[i], size=clean_xyz.shape[0])
        corrupted[:, i] = scale_xyz[i] * clean_xyz[:, i] + bias_xyz[i] + noise
    return corrupted


def estimate_bias_only(clean, corrupted):
    return float(np.mean(corrupted - clean))


def estimate_scale_and_bias(clean, corrupted):
    """
    Least-squares estimate of:
        corrupted = scale * clean + bias + noise
    """
    x = np.asarray(clean, dtype=float)
    y = np.asarray(corrupted, dtype=float)

    A = np.column_stack([x, np.ones_like(x)])
    theta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    scale_hat = float(theta[0])
    bias_hat = float(theta[1])
    return scale_hat, bias_hat


def estimate_all_axes(clean_xyz, corr_xyz, eps=1e-8):
    scale_hat = np.ones(3, dtype=float)
    bias_hat = np.zeros(3, dtype=float)

    for i in range(3):
        clean_axis = clean_xyz[:, i]
        corr_axis = corr_xyz[:, i]

        if np.std(clean_axis) < eps:
            # Not enough excitation to estimate scale reliably
            scale_hat[i] = 1.0
            bias_hat[i] = np.mean(corr_axis - clean_axis)
        else:
            scale_hat[i], bias_hat[i] = estimate_scale_and_bias(clean_axis, corr_axis)

    return scale_hat, bias_hat


def correct_sensor_errors(corrupted_xyz, scale_hat_xyz, bias_hat_xyz, eps=1e-8):
    corrected = np.zeros_like(corrupted_xyz)

    for i in range(3):
        if abs(scale_hat_xyz[i]) < eps:
            corrected[:, i] = corrupted_xyz[:, i] - bias_hat_xyz[i]
        else:
            corrected[:, i] = (corrupted_xyz[:, i] - bias_hat_xyz[i]) / scale_hat_xyz[i]

    return corrected


# ---------------------------------------------------------------------
# RECONSTRUCTION
# ---------------------------------------------------------------------
def yaw_to_rotation_body_to_world(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])


def reconstruct_trajectory_from_imu(dt, accel_body, gyro_body, pos0, vel0, yaw0):
    """
    Reconstruct trajectory using corrected IMU.

    Steps:
    1. Integrate gyro z to get yaw
    2. Rotate accel body -> world
    3. Add gravity back to recover world acceleration
    4. Integrate acceleration -> velocity
    5. Integrate velocity -> position

    Assumptions:
      - planar yaw-only orientation
      - roll = pitch = 0
    """
    n = accel_body.shape[0]

    yaw = np.zeros(n, dtype=float)
    vel = np.zeros((n, 3), dtype=float)
    pos = np.zeros((n, 3), dtype=float)
    acc_world = np.zeros((n, 3), dtype=float)

    yaw[0] = yaw0
    vel[0] = vel0
    pos[0] = pos0

    g_world = np.array([0.0, 0.0, -GRAVITY])

    for k in range(1, n):
        yaw[k] = yaw[k - 1] + gyro_body[k - 1, 2] * dt

        R_wb = yaw_to_rotation_body_to_world(yaw[k])
        spec_force_world = R_wb @ accel_body[k]
        acc_world[k] = spec_force_world + g_world

        vel[k] = vel[k - 1] + acc_world[k] * dt
        pos[k] = pos[k - 1] + vel[k - 1] * dt + 0.5 * acc_world[k] * dt * dt

    return yaw, vel, pos, acc_world


# ---------------------------------------------------------------------
# METRICS / REPORT
# ---------------------------------------------------------------------
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def build_report(clean_accel, corr_accel, clean_gyro, corr_gyro,
                 accel_scale_hat, accel_bias_hat, gyro_scale_hat, gyro_bias_hat):
    rows = []

    sensor_blocks = [
        ("accelerometer", clean_accel, corr_accel, ACCEL_BIAS, ACCEL_SCALE, accel_bias_hat, accel_scale_hat),
        ("gyroscope", clean_gyro, corr_gyro, GYRO_BIAS, GYRO_SCALE, gyro_bias_hat, gyro_scale_hat),
    ]

    for sensor_name, clean_block, corr_block, true_bias, true_scale, est_bias_vec, est_scale_vec in sensor_blocks:
        for axis_idx, axis_name in enumerate(("x", "y", "z")):
            clean = clean_block[:, axis_idx]
            corr = corr_block[:, axis_idx]

            bias_only_hat = estimate_bias_only(clean, corr)

            rows.append({
                "sensor": sensor_name,
                "axis": axis_name,
                "true_bias": true_bias[axis_idx],
                "estimated_bias_only": bias_only_hat,
                "true_scale": true_scale[axis_idx],
                "estimated_scale": est_scale_vec[axis_idx],
                "estimated_bias": est_bias_vec[axis_idx],
                "bias_error": est_bias_vec[axis_idx] - true_bias[axis_idx],
                "scale_error": est_scale_vec[axis_idx] - true_scale[axis_idx],
            })

    return pd.DataFrame(rows)


def print_summary(report_df, metrics):
    print("\n" + "=" * 76)
    print("FULL NON-STATIC IMU CALIBRATION + RECONSTRUCTION REPORT")
    print("=" * 76)

    for sensor in ("accelerometer", "gyroscope"):
        print(f"\n{sensor.upper()}")
        print("-" * 76)
        sub = report_df[report_df["sensor"] == sensor]
        for _, row in sub.iterrows():
            print(
                f"axis {row['axis']}: "
                f"true bias = {row['true_bias']:+.6e}, "
                f"est bias = {row['estimated_bias']:+.6e}, "
                f"true scale = {row['true_scale']:+.6f}, "
                f"est scale = {row['estimated_scale']:+.6f}"
            )

    print("\nRECONSTRUCTION METRICS")
    print("-" * 76)
    for key, value in metrics.items():
        print(f"{key}: {value:.6e}")


# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------
def plot_imu_signals(t, accel_clean, accel_corr, accel_corrected,
                     gyro_clean, gyro_corr, gyro_corrected):
    fig, axs = plt.subplots(2, 3, figsize=(16, 7))
    axes = ["x", "y", "z"]

    for i in range(3):
        axs[0, i].plot(t, accel_clean[:, i], label="clean", linewidth=1)
        axs[0, i].plot(t, accel_corr[:, i], label="corrupted", linewidth=1)
        axs[0, i].plot(t, accel_corrected[:, i], label="corrected", linewidth=1)
        axs[0, i].set_title(f"Accel {axes[i]}")
        axs[0, i].set_xlabel("Time [s]")
        axs[0, i].set_ylabel("m/s²")
        axs[0, i].grid()

    for i in range(3):
        axs[1, i].plot(t, gyro_clean[:, i], label="clean", linewidth=1)
        axs[1, i].plot(t, gyro_corr[:, i], label="corrupted", linewidth=1)
        axs[1, i].plot(t, gyro_corrected[:, i], label="corrected", linewidth=1)
        axs[1, i].set_title(f"Gyro {axes[i]}")
        axs[1, i].set_xlabel("Time [s]")
        axs[1, i].set_ylabel("rad/s")
        axs[1, i].grid()

    axs[0, 0].legend()
    axs[1, 0].legend()
    plt.tight_layout()
    plt.show()


def plot_error(t, clean, corr, corrected, title):
    err_corr = corr - clean
    err_corrected = corrected - clean

    plt.figure(figsize=(10, 4))
    for i, axis in enumerate(["x", "y", "z"]):
        plt.plot(t, err_corr[:, i], label=f"{axis} corrupted")
        plt.plot(t, err_corrected[:, i], linestyle="--", label=f"{axis} corrected")
    plt.title(f"{title} error")
    plt.xlabel("Time [s]")
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
    plt.show()


def plot_trajectory(true_pos, recon_pos):
    plt.figure(figsize=(7, 6))
    plt.plot(true_pos[:, 0], true_pos[:, 1], label="true trajectory")
    plt.plot(recon_pos[:, 0], recon_pos[:, 1], label="reconstructed trajectory")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Trajectory: true vs reconstructed")
    plt.axis("equal")
    plt.grid()
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    # straight_line = 1
    # circle = 2
    # ellipse = 3
    # infinity = 4
    c_type = 3

    stp = [1.5, 2.5]
    edp = [3.5, 2.5]
    total_time = 40.0
    num_points = 500

    print("1) Building non-static simple trajectory...")
    t, dt, pos_true = build_time_parameterized_trajectory(
        stp, edp, num_points=num_points, c_type=c_type, total_time=total_time
    )

    print("2) Computing clean synthetic IMU from physics...")
    vel_true, acc_world_true, heading_true, gyro_clean, accel_clean = compute_clean_imu_from_positions(dt, pos_true)

    rng = np.random.default_rng(SEED)

    print("3) Injecting known accelerometer / gyroscope scale, bias, and noise...")
    accel_corr = inject_sensor_errors(
        accel_clean, ACCEL_SCALE, ACCEL_BIAS, ACCEL_NOISE_STD, rng
    )
    gyro_corr = inject_sensor_errors(
        gyro_clean, GYRO_SCALE, GYRO_BIAS, GYRO_NOISE_STD, rng
    )

    print("4) Estimating scale and bias with least squares...")
    accel_scale_hat, accel_bias_hat = estimate_all_axes(accel_clean, accel_corr)
    gyro_scale_hat, gyro_bias_hat = estimate_all_axes(gyro_clean, gyro_corr)

    print("5) Correcting corrupted IMU using estimated parameters...")
    accel_corrected = correct_sensor_errors(accel_corr, accel_scale_hat, accel_bias_hat)
    gyro_corrected = correct_sensor_errors(gyro_corr, gyro_scale_hat, gyro_bias_hat)

    print("6) Reconstructing trajectory from corrected IMU...")
    yaw_recon, vel_recon, pos_recon, acc_world_recon = reconstruct_trajectory_from_imu(
        dt=dt,
        accel_body=accel_corrected,
        gyro_body=gyro_corrected,
        pos0=pos_true[0],
        vel0=vel_true[0],
        yaw0=heading_true[0],
    )

    print("7) Building report and validation metrics...")
    report_df = build_report(
        accel_clean, accel_corr, gyro_clean, gyro_corr,
        accel_scale_hat, accel_bias_hat, gyro_scale_hat, gyro_bias_hat
    )

    metrics = {
        "rmse_accel_corrupted_vs_clean": rmse(accel_corr, accel_clean),
        "rmse_accel_corrected_vs_clean": rmse(accel_corrected, accel_clean),
        "rmse_gyro_corrupted_vs_clean": rmse(gyro_corr, gyro_clean),
        "rmse_gyro_corrected_vs_clean": rmse(gyro_corrected, gyro_clean),
        "rmse_position_reconstructed_vs_true": rmse(pos_recon, pos_true),
        "rmse_velocity_reconstructed_vs_true": rmse(vel_recon, vel_true),
        "rmse_yaw_reconstructed_vs_true": rmse(yaw_recon, heading_true),
    }

    df = pd.DataFrame({
        "t": t,

        "pos_x_true": pos_true[:, 0],
        "pos_y_true": pos_true[:, 1],
        "pos_z_true": pos_true[:, 2],

        "vel_x_true": vel_true[:, 0],
        "vel_y_true": vel_true[:, 1],
        "vel_z_true": vel_true[:, 2],

        "yaw_true": heading_true,

        "acc_x_clean": accel_clean[:, 0],
        "acc_y_clean": accel_clean[:, 1],
        "acc_z_clean": accel_clean[:, 2],

        "gyro_x_clean": gyro_clean[:, 0],
        "gyro_y_clean": gyro_clean[:, 1],
        "gyro_z_clean": gyro_clean[:, 2],

        "acc_x_corrupted": accel_corr[:, 0],
        "acc_y_corrupted": accel_corr[:, 1],
        "acc_z_corrupted": accel_corr[:, 2],

        "gyro_x_corrupted": gyro_corr[:, 0],
        "gyro_y_corrupted": gyro_corr[:, 1],
        "gyro_z_corrupted": gyro_corr[:, 2],

        "acc_x_corrected": accel_corrected[:, 0],
        "acc_y_corrected": accel_corrected[:, 1],
        "acc_z_corrected": accel_corrected[:, 2],

        "gyro_x_corrected": gyro_corrected[:, 0],
        "gyro_y_corrected": gyro_corrected[:, 1],
        "gyro_z_corrected": gyro_corrected[:, 2],

        "yaw_reconstructed": yaw_recon,

        "vel_x_reconstructed": vel_recon[:, 0],
        "vel_y_reconstructed": vel_recon[:, 1],
        "vel_z_reconstructed": vel_recon[:, 2],

        "pos_x_reconstructed": pos_recon[:, 0],
        "pos_y_reconstructed": pos_recon[:, 1],
        "pos_z_reconstructed": pos_recon[:, 2],
    })

    print("\nInjected values:")
    print("Accelerometer bias:", ACCEL_BIAS)
    print("Gyroscope bias:", GYRO_BIAS)
    print("Accelerometer scale:", ACCEL_SCALE)
    print("Gyroscope scale:", GYRO_SCALE)
    print("Accelerometer white noise std:", ACCEL_NOISE_STD)
    print("Gyroscope white noise std:", GYRO_NOISE_STD)

    print("\nEstimated values:")
    print("Accelerometer bias hat:", accel_bias_hat)
    print("Gyroscope bias hat:", gyro_bias_hat)
    print("Accelerometer scale hat:", accel_scale_hat)
    print("Gyroscope scale hat:", gyro_scale_hat)

    print_summary(report_df, metrics)

    df.to_csv(OUTPUT_DATA_CSV, index=False)
    report_df.to_csv(OUTPUT_REPORT_CSV, index=False)

    print(f"\nSaved full dataset to: {OUTPUT_DATA_CSV}")
    print(f"Saved calibration report to: {OUTPUT_REPORT_CSV}")

    plot_imu_signals(
        t, accel_clean, accel_corr, accel_corrected,
        gyro_clean, gyro_corr, gyro_corrected
    )
    plot_error(t, accel_clean, accel_corr, accel_corrected, "Accelerometer")
    plot_error(t, gyro_clean, gyro_corr, gyro_corrected, "Gyroscope")
    plot_trajectory(pos_true, pos_recon)


if __name__ == "__main__":
    main()
