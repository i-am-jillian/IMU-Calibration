#imu simulator
import numpy as np

# ---------------------------------------------------------------------------
# Constants (from imusim/environment/gravity.py)
# ---------------------------------------------------------------------------
STANDARD_GRAVITY = 9.81          # m/s²
GRAVITY_VECTOR   = np.array([0.0, 0.0, STANDARD_GRAVITY])   # z-down [0,0,9.81]

# ---------------------------------------------------------------------------
# Quaternion helpers (minimal port of imusim/maths/quaternions.py)
# ---------------------------------------------------------------------------
def quat_rotate_frame(q, v):
    """
    Rotate vector v from the world frame into the sensor (body) frame.
    converts quaternion into matrix form
    """
    w, x, y, z = q #q= scalar(how much rotation there is), vector, vector, vector(direction of rotation)
    # rotation matrix from q (rotational vector derived from quaternion)
    R = np.array([
        [1 - 2*(y*y + z*z),  2*(x*y + w*z),       2*(x*z - w*y)],
        [2*(x*y - w*z),       1 - 2*(x*x + z*z),   2*(y*z + w*x)],
        [2*(x*z + w*y),       2*(y*z - w*x),        1 - 2*(x*x + y*y)],
    ])
    return R @ v #R*v


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------
class StaticTrajectory:
    """
    At rest:  zero velocity, zero acceleration, zero rotational
    velocity.  Orientation is given by a unit quaternion [w, x, y, z].
    so velocity, acce, and ang vel =0 but it can still have an orientation
    """

    def __init__(self, rotation=None, position=None):
        """
        Parameters
        ----------
        rotation : array-like shape (4), [w, x, y, z]
            Constant orientation quaternion (default: identity).
        position : array-like shape (3,) (3d vector)
            Constant position in world frame (default: origin).(if not given [0,0,0])
        """
        self.rotation = np.array(rotation if rotation is not None
                                 else [1.0, 0.0, 0.0, 0.0])#if no rotation given, use [1,0,0,0]
        
        self.rotation /= np.linalg.norm(self.rotation) #q=q/|q| normalized quaternion to guarantee valid rotation
        self.position = np.array(position if position is not None
                                 else [0.0, 0.0, 0.0])#if position provide it use it, if not [0,0,0]
        self.accel_bias_instability = accel_bias_instability
        self.gyro_bias_instability = gyro_bias_instability

    def acceleration(self):
        """Linear acceleration in world frame (zero for static object)."""
        return np.zeros(3)#no linear acc bc imu is not moving or accelerating

    def rotational_velocity(self):
        """Angular velocity in world frame (zero for static object)."""
        return np.zeros(3)#same here
    

# ---------------------------------------------------------------------------
# Sensor models
# ---------------------------------------------------------------------------
class IdealAccelerometer:

    def measure(self, trajectory, N):
        """
        Return N ideal accelerometer samples (3×N array, m/s²).
        """
        a_world = trajectory.acceleration() - GRAVITY_VECTOR
        a_body  = quat_rotate_frame(trajectory.rotation, a_world)
        return np.tile(a_body[:, None], (1, N))


class IdealGyroscope:

    def measure(self, trajectory, N):
        """Return N ideal gyroscope samples (3×N array, rad/s)."""
        omega_world = trajectory.rotational_velocity()
        omega_body  = quat_rotate_frame(trajectory.rotation, omega_world)
        return np.tile(omega_body[:, None], (1, N))


class NoisyAccelerometer:
    """
    Adds Gaussian noise with std = noise_std to each sample.
    This is the 'white noise' / angle random walk term on the Allan plot.
    """

    def __init__(self, noise_std, rng=None):
        self._ideal = IdealAccelerometer()
        self.noise_std = noise_std #stores noise level (standard deviation)
        self.rng = rng or np.random.default_rng() #creates randomness

    def measure(self, trajectory, N):
        true = self._ideal.measure(trajectory, N)
        noise = self.rng.normal(scale=self.noise_std, size=(3, N)) #adds noise 3 axis n times
    #each axis gets its own noise value at every time step

class NoisyGyroscope:
#noise_std sets the width of the cloud, and rng.normal picks a randome value (noise) from that cloud
    def __init__(self, noise_std, rng=None):
        """
        Parameters
        ----------
        noise_std : float
            1-sigma white noise standard deviation (rad/s).
        rng : np.random.Generator, optional
        """
        self._ideal = IdealGyroscope()
        self.noise_std = noise_std
        self.rng = rng or np.random.default_rng()

    def measure(self, trajectory, N):
        true = self._ideal.measure(trajectory, N)
        noise = self.rng.normal(scale=self.noise_std, size=(3, N))
        return true + noise #0+noise


class RealisticIMU: #includes white noise, fixed bias, bias drift, scale errors and cross-axis errors
    # Typical parameter ranges from IMUSim MMA7260Q / ADXRS300 documentation
    ACCEL_SCALE_ERROR_STD   = 0.075 / 3        # ±7.5 % (3σ)
    ACCEL_CROSS_AXIS_MAX    = 0.05             # ±5 %
    GYRO_SCALE_ERROR_STD    = 0.08  / 3        # ±8 % (3σ)
    GYRO_CROSS_AXIS_STD     = 0.05  / 3

    def __init__(
        self, 
        fs=100.0, 
        accel_noise_std=0.005,        # ~5 mg white noise
        gyro_noise_std=0.003,         # ~0.17 °/s white noise
        accel_bias_rw_std=1e-4,       # random walk diffusion per sample
        gyro_bias_rw_std=5e-5,
        accel_fixed_bias=None,
        gyro_fixed_bias=None,
        seed=None,
    ):
    def _flicker_noise(self, N, target_B):
        """
        Generate approximate 1/f noise per axis.
        target_B is the bias-instability coefficient you want the Allan analysis to recover.
        """
        if target_B <= 0:
            return np.zeros((3, N))

        out = np.zeros((3, N))
        scf_b = np.sqrt(2 * np.log(2) / np.pi)

        for ax in range(3):
            freqs = np.fft.rfftfreq(N, d=1.0 / self.fs)
            reqs[0] = freqs[1]  # avoid divide-by-zero

            white_spec = (
                self.rng.normal(size=len(freqs))
                + 1j * self.rng.normal(size=len(freqs))
            )

            # 1/sqrt(f) amplitude -> 1/f PSD
            spec = white_spec / np.sqrt(freqs)
            x = np.fft.irfft(spec, n=N)
            x = x - np.mean(x)
            x = x / np.std(x)

            # crude scaling: make the flat Allan level near scf_b * B
            x = x * (scf_b * target_B)
            out[ax] = x

        return out

        self.fs = fs 
        self.rng = np.random.default_rng(seed) #creates random generator

        # --- accelerometer imperfections (from MMA7260Q model) ---------------
        a_scale = 1.0 + self.rng.normal(scale=self.ACCEL_SCALE_ERROR_STD, size=3) #creates 3 randome scale factors one for each axis
        a_cross = np.eye(3) #starts with identity matrix
        for i, j in [(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)]:
            a_cross[i, j] = self.rng.normal(scale=self.ACCEL_CROSS_AXIS_MAX / 3) #puts random values in those off diagonal spots
        self.accel_transform = np.diag(a_scale) @ a_cross

        if accel_fixed_bias is not None:
            self.accel_fixed_bias = np.asarray(accel_fixed_bias, dtype=float)
        else:
            # Random fixed offset: a few mg range
            self.accel_fixed_bias = self.rng.normal(scale=0.02, size=3)

        self.accel_noise_std   = accel_noise_std
        self.accel_bias_rw_std = accel_bias_rw_std

        # --- gyroscope imperfections (from ADXRS300 model) -------------------
        g_scale = 1.0 + self.rng.normal(scale=self.GYRO_SCALE_ERROR_STD, size=3)
        g_cross = np.eye(3)
        for i, j in [(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)]:
            g_cross[i, j] = self.rng.normal(scale=self.GYRO_CROSS_AXIS_STD)
        self.gyro_transform = np.diag(g_scale) @ g_cross

        if gyro_fixed_bias is not None:
            self.gyro_fixed_bias = np.asarray(gyro_fixed_bias, dtype=float)
        else:
            # Random fixed offset: a few m°/s range
            self.gyro_fixed_bias = self.rng.normal(scale=0.005, size=3)

        self.gyro_noise_std   = gyro_noise_std
        self.gyro_bias_rw_std = gyro_bias_rw_std

    # -------------------------------------------------------------------------
    def _bias_random_walk(self, N, std_per_sample):
        """
        Integrated Gaussian noise — bias random walk (Allan slope +0.5).
        Each axis is an independent Wiener process.
        std_per_sample = sigma_b / sqrt(fs)
        """
        increments = self.rng.normal(scale=std_per_sample, size=(3, N))
        return np.cumsum(increments, axis=1)

    # -------------------------------------------------------------------------
    def simulate(self, trajectory, duration):
        """
        Simulate a static IMU for `duration` seconds at `self.fs` Hz.

        Returns
        -------
        t       : np.ndarray (N,)          sample timestamps (s)
        accel   : np.ndarray (3, N)        accelerometer output (m/s²)
        gyro    : np.ndarray (3, N)        gyroscope output    (rad/s)
        truth   : dict with ground-truth values and injected bias components
        """
        N  = int(duration * self.fs)
        dt = 1.0 / self.fs
        t  = np.arange(N) * dt

        # --- true values (static: accel = −g in body frame, gyro = 0) -------
        ideal_accel = IdealAccelerometer().measure(trajectory, N)   # (3,N)
        ideal_gyro  = IdealGyroscope().measure(trajectory, N)       # (3,N)

        # --- apply scale / cross-axis (deterministic sensor imperfection) ----
        accel_true_scaled = self.accel_transform @ ideal_accel
        gyro_true_scaled  = self.gyro_transform  @ ideal_gyro

        # --- noise components -------------------------------------------------
        accel_white = self.rng.normal(scale=self.accel_noise_std, size=(3, N))
        gyro_white  = self.rng.normal(scale=self.gyro_noise_std,  size=(3, N))

        accel_rw = self._bias_random_walk(N, self.accel_bias_rw_std)
        gyro_rw  = self._bias_random_walk(N, self.gyro_bias_rw_std)

        # --- fixed bias -------------------------------------------------------
        accel_fixed = self.accel_fixed_bias[:, None] * np.ones((1, N))
        gyro_fixed  = self.gyro_fixed_bias[:, None]  * np.ones((1, N))

        # --- total measurement ------------------------------------------------
        accel_out = accel_true_scaled + accel_fixed + accel_white + accel_rw
        gyro_out  = gyro_true_scaled  + gyro_fixed  + gyro_white  + gyro_rw

        truth = dict(
            ideal_accel      = ideal_accel,
            ideal_gyro       = ideal_gyro,
            accel_fixed_bias = self.accel_fixed_bias.copy(),
            gyro_fixed_bias  = self.gyro_fixed_bias.copy(),
            accel_white      = accel_white,
            gyro_white       = gyro_white,
            accel_bias_rw    = accel_rw,
            gyro_bias_rw     = gyro_rw,
        )

        return t, accel_out, gyro_out, truth