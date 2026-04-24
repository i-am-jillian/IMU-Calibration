"""
generate_imu_data_fixed_v2.py
=============================
Generates synthetic IMU data for Allan-deviation validation.

These defaults are chosen so bias instability is actually visible
instead of being buried under the random-walk term.

Run:
    python generate_imu_data_fixed_v2.py --duration 7200 --fs 100 --out data/test_imu.csv
"""

import argparse
import os
import numpy as np
import pandas as pd

from Imusim_fixed_corrected import StaticTrajectory, RealisticIMU


parser = argparse.ArgumentParser(description="Generate synthetic IMU data for testing")
parser.add_argument("--duration", type=float, default=7200.0)
parser.add_argument("--fs", type=float, default=100.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", type=str, default="data/test_imu.csv")
args = parser.parse_args()

out_dir = os.path.dirname(args.out)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

trajectory = StaticTrajectory(
    rotation=np.array([1.0, 0.0, 0.0, 0.0]),
    position=np.array([0.0, 0.0, 0.0]),
)

# Lower random walk and stronger bias instability so the plateau is measurable.
imu = RealisticIMU(
    fs=args.fs,
    accel_noise_std=1e-2,
    gyro_noise_std=3e-3,
    accel_bias_rw_std=4e-6,
    gyro_bias_rw_std=3e-6,
    accel_bias_instability=4e-4,
    gyro_bias_instability=1.5e-4,
    accel_fixed_bias=np.array([0.08, -0.06, 0.03]),
    gyro_fixed_bias=np.array([0.01, -0.015, 0.008]),
    seed=args.seed,
)

t, accel, gyro, truth = imu.simulate(trajectory, args.duration)

df = pd.DataFrame({
    "t": t,
    "ax": accel[0], "ay": accel[1], "az": accel[2],
    "gx": gyro[0],  "gy": gyro[1],  "gz": gyro[2],
})

df.to_csv(args.out, index=False)
print(f"Saved synthetic test data to: {args.out}")

print("\nInjected terms used for validation:")
print("Accelerometer fixed bias:", truth["accel_fixed_bias"])
print("Gyroscope fixed bias:", truth["gyro_fixed_bias"])
print("Accel white noise std:", imu.accel_noise_std)
print("Gyro white noise std:", imu.gyro_noise_std)
print("Accel bias RW std/sample:", imu.accel_bias_rw_std)
print("Gyro bias RW std/sample:", imu.gyro_bias_rw_std)
print("Accel bias instability coeff B:", imu.accel_bias_instability)
print("Gyro bias instability coeff B:", imu.gyro_bias_instability)
print("Expected accel RW coeff K ≈", imu.accel_bias_rw_std * np.sqrt(imu.fs))
print("Expected gyro  RW coeff K ≈", imu.gyro_bias_rw_std * np.sqrt(imu.fs))