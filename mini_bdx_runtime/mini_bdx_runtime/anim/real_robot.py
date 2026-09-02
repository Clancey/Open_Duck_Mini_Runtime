"""On-robot adapter implementing :class:`RobotInterface` (plan §6.1, §6.5).

This is the ONLY module in the ``anim`` subpackage that touches physical
devices, and it imports them **lazily** (inside ``connect``) so the rest of the
package — the FSM, safety monitor and controller — stays importable on a laptop
or CI with no ``board`` / ``rustypot`` / ``pygame`` present. The unit tests use
:class:`~mini_bdx_runtime.anim.hardware.MockRobot` instead.

Mapping to the existing runtime devices (unchanged, we only adapt them):

* joints/gains/torque -> ``rustypot_position_hwi.HWI``
* tilt                -> ``raw_imu.Imu`` accelerometer gravity estimate (the
  runtime IMU exposes only gyro+accel, no orientation quaternion, so we derive
  the tilt-from-vertical angle here and document it)
* antennas            -> ``antennas.Antennas`` (PWM D13 left +1 / D12 right -1)
* eyes                -> ``eyes.Eyes`` (its auto-blink thread is stopped so the
  animation eye track has exclusive control)
* sounds              -> ``sounds.Sounds``
* projector           -> ``projector.Projector``

TEMPERATURE / CURRENT: ``rustypot==0.1.0`` does not expose the STS3215 Present
Temperature / Present Current registers, so ``temperatures_c`` / ``currents_a``
stay ``None`` and the ThermalManager degrades to the time-based duty limit. This
is the single hardware-unverifiable gap flagged in the report.
"""

import time
from typing import List, Optional

import numpy as np

from .hardware import RobotInterface, SensorSnapshot, OperatorInput, N_DOFS

# Bus joint order (matches HWI.joints insertion order and open_duck_anim HW14).
_JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]


def _tilt_from_accel(accel: np.ndarray) -> float:
    """Estimate tilt-from-vertical (rad) from the accelerometer gravity vector.

    With the robot roughly static, the accelerometer measures gravity. The tilt
    magnitude is the angle between the measured acceleration and the body Z axis
    (the upright gravity direction). This is only valid when linear acceleration
    is small (true in DOCK_DEMO / quiescent ARMING); it is NOT a substitute for a
    filtered orientation estimate during dynamic walking. Documented so the
    caller knows the limitation.
    """
    a = np.asarray(accel, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n < 1e-6:
        return 0.0
    cos_tilt = float(np.clip(abs(a[2]) / n, 0.0, 1.0))
    return float(np.arccos(cos_tilt))


class RealRobot(RobotInterface):
    """Real hardware adapter. Call :meth:`connect` before use."""

    def __init__(self, duck_config=None, usb_port: str = "/dev/ttyACM0",
                 sound_directory: str = "./", enable_sounds: bool = True,
                 enable_projector: bool = True, enable_eyes: bool = True):
        self.usb_port = usb_port
        self._duck_config = duck_config
        self._sound_directory = sound_directory
        self._enable_sounds = enable_sounds
        self._enable_projector = enable_projector
        self._enable_eyes = enable_eyes

        self.hwi = None
        self.imu = None
        self.antennas = None
        self.eyes = None
        self.sounds = None
        self.projector = None
        self._connected = False

    def connect(self) -> None:
        """Lazily import and open every device. Raises if hardware is absent."""
        from mini_bdx_runtime.rustypot_position_hwi import HWI
        from mini_bdx_runtime.raw_imu import Imu
        from mini_bdx_runtime.antennas import Antennas
        from mini_bdx_runtime.duck_config import DuckConfig

        cfg = self._duck_config or DuckConfig()
        self.hwi = HWI(cfg, self.usb_port)
        self.imu = Imu(sampling_freq=50)
        self.antennas = Antennas()
        if self._enable_eyes:
            from mini_bdx_runtime.eyes import Eyes
            self.eyes = Eyes()
            # Take exclusive control of the eye pins: stop the auto-blink thread
            # so the animation eye track is authoritative.
            try:
                self.eyes.stop()
            except Exception:
                pass
        if self._enable_sounds:
            from mini_bdx_runtime.sounds import Sounds
            self.sounds = Sounds(volume=1.0, sound_directory=self._sound_directory)
        if self._enable_projector:
            from mini_bdx_runtime.projector import Projector
            self.projector = Projector()
        self._connected = True

    # -- read ------------------------------------------------------------------
    def read(self, operator: OperatorInput) -> Optional[SensorSnapshot]:
        pos = self.hwi.get_present_positions()
        if pos is None or len(pos) != N_DOFS:
            return None
        vel = self.hwi.get_present_velocities()
        if vel is None or len(vel) != N_DOFS:
            vel = np.zeros(N_DOFS, dtype=np.float64)
        try:
            imu_data = self.imu.get_data()
            accel = np.asarray(imu_data["accelero"], dtype=np.float64)
            gyro = np.asarray(imu_data["gyro"], dtype=np.float64)
        except Exception:
            accel = np.zeros(3)
            gyro = np.zeros(3)
        tilt = _tilt_from_accel(accel)
        return SensorSnapshot(
            t_monotonic=time.monotonic(),
            joint_positions=np.asarray(pos, dtype=np.float64),
            joint_velocities=np.asarray(vel, dtype=np.float64),
            tilt_rad=tilt,
            # No dedicated foot-contact sensor on this build; DOCK_DEMO holds the
            # legs, so we report "both feet" (the dock supports the robot). This
            # is only consumed by the STAND/dock-handoff guard, which the demo
            # does not exercise. Flagged for hardware review.
            feet_contacts=np.array([1.0, 1.0]),
            gyro=gyro,
            accelero=accel,
            temperatures_c=None,   # not exposed by rustypot 0.1.0
            currents_a=None,       # not exposed by rustypot 0.1.0
            operator=operator,
        )

    # -- write -----------------------------------------------------------------
    def set_gains(self, kps: np.ndarray, kds: np.ndarray) -> None:
        self.hwi.set_kps(np.asarray(kps, dtype=np.float64))
        self.hwi.set_kds(np.asarray(kds, dtype=np.float64))

    def set_joint_targets(self, targets: np.ndarray) -> None:
        t = np.asarray(targets, dtype=np.float64)
        joints_positions = {name: float(t[i]) for i, name in enumerate(_JOINT_NAMES)}
        self.hwi.set_position_all(joints_positions)

    def torque_off(self) -> None:
        self.hwi.turn_off()

    def set_antennas(self, left_norm: float, right_norm: float) -> None:
        if self.antennas is None:
            return
        self.antennas.set_position_left(float(left_norm))
        self.antennas.set_position_right(float(right_norm))

    def set_eyes(self, state: int) -> None:
        if self.eyes is None:
            return
        try:
            self.eyes._set_eyes(bool(state))
        except Exception:
            pass

    def play_sound(self, name: str) -> None:
        if self.sounds is None:
            return
        # Accept bare names or explicit filenames.
        cand = name if name.endswith(".wav") else (name + ".wav")
        try:
            self.sounds.play(cand)
        except Exception:
            pass

    def set_projector(self, on: bool) -> None:
        if self.projector is None:
            return
        try:
            if bool(on) != bool(self.projector.on):
                self.projector.switch()
        except Exception:
            pass

    def shutdown_show(self) -> None:
        self.set_antennas(0.0, 0.0)
        self.set_projector(False)

    def close(self) -> None:
        for dev, meth in ((self.antennas, "stop"), (self.eyes, "stop"),
                          (self.projector, "stop")):
            if dev is not None:
                try:
                    getattr(dev, meth)()
                except Exception:
                    pass
