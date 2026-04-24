import numpy as np

STANDARD_GRAVITY = 9.81
GRAVITY_VECTOR = np.array([0.0, 0.0, STANDARD_GRAVITY])


def quat_rotate_frame(q, v):
    """
    Rotate vector v from the world frame into the sensor/body frame.
    q is [w, x, y, z].
    """
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y + w * z),     2 * (x * z - w * y)],
        [2 * (x * y - w * z),     1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
        [2 * (x * z + w * y),     2 * (y * z - w * x),     1 - 2 * (x * x + y * y)],
    ])
    return R @ v


class StaticTrajectory:
    """
    Static body with fixed orientation and position.
    """

    def __init__(self, rotation=None, position=None):
        self.rotation = np.array(
            rotation if rotation is not None else [1.0, 0.0, 0.0, 0.0],
            dtype=float,
        )
        self.rotation /= np.linalg.norm(self.rotation)
        self.position = np.array(
            position if position is not None else [0.0, 0.0, 0.0],
            dtype=float,
        )

    def acceleration(self):
        return np.zeros(3)

    def rotational_velocity(self):
        return np.zeros(3)


class IdealAccelerometer:
    def measure(self, trajectory, N):
        a_world = trajectory.acceleration() - GRAVITY_VECTOR
        a_body = quat_rotate_frame(trajectory.rotation, a_world)
        return np.tile(a_body[:, None], (1, N))


class IdealGyroscope:
    def measure(self, trajectory, N):
        omega_world = trajectory.rotational_velocity()
        omega_body = quat_rotate_frame(trajectory.rotation, omega_world)
        return np.tile(omega_body[:, None], (1, N))


class NoisyAccelerometer:
    def __init__(self, noise_std, rng=None):
        self._ideal = IdealAccelerometer()
        self.noise_std = float(noise_std)
        self.rng = rng or np.random.default_rng()

    def measure(self, trajectory, N):
        true = self._ideal.measure(trajectory, N)
        noise = self.rng.normal(scale=self.noise_std, size=(3, N))
        return true + noise


class NoisyGyroscope:
    def __init__(self, noise_std, rng=None):
        self._ideal = IdealGyroscope()
        self.noise_std = float(noise_std)
        self.rng = rng or np.random.default_rng()

    def measure(self, trajectory, N):
        true = self._ideal.measure(trajectory, N)
        noise = self.rng.normal(scale=self.noise_std, size=(3, N))
        return true + noise


class RealisticIMU:
    """
    Static IMU model with:
      - deterministic scale/cross-axis errors
      - fixed bias
      - white noise
      - bias random walk
      - optional flicker (1/f) bias noise to produce bias instability
    """

    ACCEL_SCALE_ERROR_STD = 0.075 / 3
    ACCEL_CROSS_AXIS_MAX = 0.05
    GYRO_SCALE_ERROR_STD = 0.08 / 3
    GYRO_CROSS_AXIS_STD = 0.05 / 3

    def __init__(
        self,
        fs=100.0,
        accel_noise_std=0.005,
        gyro_noise_std=0.003,
        accel_bias_rw_std=1e-4,
        gyro_bias_rw_std=5e-5,
        accel_bias_instability=0.0,
        gyro_bias_instability=0.0,
        accel_fixed_bias=None,
        gyro_fixed_bias=None,
        seed=None,
    ):
        self.fs = float(fs)
        self.rng = np.random.default_rng(seed)

        self.accel_noise_std = float(accel_noise_std)
        self.gyro_noise_std = float(gyro_noise_std)
        self.accel_bias_rw_std = float(accel_bias_rw_std)
        self.gyro_bias_rw_std = float(gyro_bias_rw_std)
        self.accel_bias_instability = float(accel_bias_instability)
        self.gyro_bias_instability = float(gyro_bias_instability)

        a_scale = 1.0 + self.rng.normal(scale=self.ACCEL_SCALE_ERROR_STD, size=3)
        a_cross = np.eye(3)
        for i, j in [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]:
            a_cross[i, j] = self.rng.normal(scale=self.ACCEL_CROSS_AXIS_MAX / 3)
        self.accel_transform = np.diag(a_scale) @ a_cross

        g_scale = 1.0 + self.rng.normal(scale=self.GYRO_SCALE_ERROR_STD, size=3)
        g_cross = np.eye(3)
        for i, j in [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]:
            g_cross[i, j] = self.rng.normal(scale=self.GYRO_CROSS_AXIS_STD)
        self.gyro_transform = np.diag(g_scale) @ g_cross

        if accel_fixed_bias is not None:
            self.accel_fixed_bias = np.asarray(accel_fixed_bias, dtype=float)
        else:
            self.accel_fixed_bias = self.rng.normal(scale=0.02, size=3)

        if gyro_fixed_bias is not None:
            self.gyro_fixed_bias = np.asarray(gyro_fixed_bias, dtype=float)
        else:
            self.gyro_fixed_bias = self.rng.normal(scale=0.005, size=3)

    @staticmethod
    def allan_deviation(data: np.ndarray, fs: float):
        N = len(data)
        dt = 1.0 / fs
        max_n = N // 3

        if max_n < 2:
            raise ValueError("Not enough samples for Allan deviation (need at least 6).")

        n_values = np.unique(np.round(np.logspace(0, np.log10(max_n), 200)).astype(int))

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

    @classmethod
    def geometric_mean_adev(cls, meas: np.ndarray, fs: float):
        per_axis = []
        tau_ref = None

        for ax in range(3):
            tau, adev = cls.allan_deviation(meas[ax], fs)
            if tau_ref is None:
                tau_ref = tau
            per_axis.append(adev)

        avg = np.exp(np.mean(np.log(per_axis), axis=0))
        return tau_ref, avg

    @staticmethod
    def estimate_bias_instability_from_adev(tau: np.ndarray, adev: np.ndarray):
        scf_b = np.sqrt(2.0 * np.log(2.0) / np.pi)

        log_tau = np.log10(tau)
        log_adev = np.log10(adev)
        slopes = np.gradient(log_adev, log_tau)

        flat_mask = (np.abs(slopes) <= 0.08) & (tau >= 5.0)

        if np.any(flat_mask):
            bi_meas = np.min(adev[flat_mask])
        else:
            mid = (tau >= max(5.0, tau[len(tau) // 4])) & (tau <= tau[len(tau) // 2])
            bi_meas = np.median(adev[mid]) if np.any(mid) else np.median(adev)

        return bi_meas / scf_b

    def _bias_random_walk(self, N, std_per_sample):
        increments = self.rng.normal(scale=std_per_sample, size=(3, N))
        return np.cumsum(increments, axis=1)

    def _flicker_noise(self, N, scale):
        """
        Generate approximate 1/f noise with user-provided amplitude scale.
        """
        if scale <= 0:
            return np.zeros((3, N))

        out = np.zeros((3, N))

        for ax in range(3):
            freqs = np.fft.rfftfreq(N, d=1.0 / self.fs)
            if len(freqs) < 2:
                return np.zeros((3, N))
            freqs[0] = freqs[1]

            white_spec = (
                self.rng.normal(size=len(freqs))
                + 1j * self.rng.normal(size=len(freqs))
            )

            spec = white_spec / np.sqrt(freqs)
            x = np.fft.irfft(spec, n=N)
            x = x - np.mean(x)

            std = np.std(x)
            if std > 0:
                x = x / std

            out[ax] = scale * x

        return out

    def _calibrate_flicker_scale(self, N, target_B, white, rw, n_iter=12):
        """
        Calibrate flicker amplitude so TOTAL stochastic signal matches target B.
        """
        if target_B <= 0:
            return np.zeros((3, N))

        raw = self._flicker_noise(N, 1.0)

        lo, hi = 0.0, 10.0 * target_B
        best = np.zeros((3, N))

        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)

            candidate = white + rw + mid * raw
            tau, adev = self.geometric_mean_adev(candidate, self.fs)
            B_mid = self.estimate_bias_instability_from_adev(tau, adev)

            best = mid * raw

            if B_mid < target_B:
                lo = mid
            else:
                hi = mid

        return best

    def simulate(self, trajectory, duration):
        N = int(duration * self.fs)
        dt = 1.0 / self.fs
        t = np.arange(N) * dt

        ideal_accel = IdealAccelerometer().measure(trajectory, N)
        ideal_gyro = IdealGyroscope().measure(trajectory, N)

        accel_true_scaled = self.accel_transform @ ideal_accel
        gyro_true_scaled = self.gyro_transform @ ideal_gyro

        accel_white = self.rng.normal(scale=self.accel_noise_std, size=(3, N))
        gyro_white = self.rng.normal(scale=self.gyro_noise_std, size=(3, N))

        accel_rw = self._bias_random_walk(N, self.accel_bias_rw_std)
        gyro_rw = self._bias_random_walk(N, self.gyro_bias_rw_std)

        accel_flicker = self._calibrate_flicker_scale(
            N, self.accel_bias_instability, accel_white, accel_rw
        )
        gyro_flicker = self._calibrate_flicker_scale(
            N, self.gyro_bias_instability, gyro_white, gyro_rw
        )

        accel_fixed = self.accel_fixed_bias[:, None] * np.ones((1, N))
        gyro_fixed = self.gyro_fixed_bias[:, None] * np.ones((1, N))

        accel_out = (
            accel_true_scaled + accel_fixed + accel_white + accel_rw + accel_flicker
        )
        gyro_out = (
            gyro_true_scaled + gyro_fixed + gyro_white + gyro_rw + gyro_flicker
        )

        truth = dict(
            ideal_accel=ideal_accel,
            ideal_gyro=ideal_gyro,
            accel_fixed_bias=self.accel_fixed_bias.copy(),
            gyro_fixed_bias=self.gyro_fixed_bias.copy(),
            accel_white=accel_white,
            gyro_white=gyro_white,
            accel_bias_rw=accel_rw,
            gyro_bias_rw=gyro_rw,
            accel_bias_instability=accel_flicker,
            gyro_bias_instability=gyro_flicker,
            accel_bias_instability_coeff=self.accel_bias_instability,
            gyro_bias_instability_coeff=self.gyro_bias_instability,
        )

        return t, accel_out, gyro_out, truth
