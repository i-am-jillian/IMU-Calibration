import numpy as np
import matlab.engine
import matlab
import pandas as pd
from pathlib import Path
from waypoints import gen_wp


def simulate(stp, edp, imu_fs=20, vMax=0.1, c_type=3, matlab_dir=None):
    waypoints = gen_wp(stp, edp, num_points=100, c_type=c_type)
    waypoints_ = np.hstack([waypoints, np.zeros((waypoints.shape[0], 1))])

    eng = matlab.engine.start_matlab()

    # Use the script's folder by default, or a folder you pass in
    if matlab_dir is None:
        matlab_dir = Path(__file__).resolve().parent
    else:
        matlab_dir = Path(matlab_dir).resolve()

    eng.addpath(str(matlab_dir), nargout=0)

    T = eng.imu_sim(
        matlab.double(waypoints_.tolist()),
        float(imu_fs),
        float(vMax),
        nargout=1
    )

    data_struct = eng.eval("table2struct(T, 'ToScalar', true)", nargout=1)
    eng.quit()

    imu_df = pd.DataFrame({
        col: np.array(data).flatten()
        for col, data in data_struct.items()
    })

    imu_df.to_csv("simulated_imu_data.csv", index=False)
    return imu_df


if __name__ == "__main__":
    stp = [1.5, 2.5]
    edp = [3.5, 2.5]
    imu_fs = 20
    vMax = 0.1

    imu_df = simulate(stp, edp, imu_fs=imu_fs, vMax=vMax, c_type=3)
    print(imu_df.head())
    print(imu_df.columns.tolist())