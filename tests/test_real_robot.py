"""Hardware-safety tests for :mod:`mini_bdx_runtime.anim.real_robot`.

These run with NO robot: every on-robot device module that ``RealRobot.connect``
imports lazily is replaced by a pure-Python spy injected into ``sys.modules``.
The point is to prove — against the spies, exactly as the plan requires — that

* constructing / connecting a :class:`RealRobot` never constructs
  :class:`Antennas` (whose real ``__init__`` drives the PWM pins), and
* connecting never torques the leg/head bus on (no ``turn_on`` / ``set_kps`` /
  ``set_position_all``), and
* the antenna PWM is energised only on the first *genuine* (non-neutral)
  ``set_antennas`` command — i.e. once the FSM has left DISARMED — and never for
  the neutral "show off" command the controller issues in BOOT/DISARMED/FAULT,
  and never at all when antennas are disabled or the attach is read-only.
"""

import sys
import types

import numpy as np
import pytest


class _SpyAntennas:
    """Stand-in for the real Antennas: counts constructions and commands.

    The real constructor writes a PWM duty cycle immediately, so a nonzero
    ``instances`` count is exactly "the antenna pins were energised".
    """

    instances = 0

    def __init__(self):
        type(self).instances += 1
        self.left = None
        self.right = None
        self.stopped = False

    def set_position_left(self, v):
        self.left = float(v)

    def set_position_right(self, v):
        self.right = float(v)

    def stop(self):
        self.stopped = True


class _SpyHWI:
    """Bus HWI spy. Records any call that would energise the servos."""

    def __init__(self, duck_config, usb_port="/dev/ttyACM0"):
        self.usb_port = usb_port
        self.actuated = []  # names of any torque-producing calls

    def turn_on(self):
        self.actuated.append("turn_on")

    def set_kps(self, kps):
        self.actuated.append("set_kps")

    def set_kds(self, kds):
        self.actuated.append("set_kds")

    def set_position_all(self, positions):
        self.actuated.append("set_position_all")

    def turn_off(self):
        self.actuated.append("turn_off")


class _SpyImu:
    def __init__(self, sampling_freq=50):
        self.sampling_freq = sampling_freq

    def get_data(self):
        return {"accelero": np.array([0.0, 0.0, 9.81]), "gyro": np.zeros(3)}


class _SpyDuckConfig:
    def __init__(self, *a, **k):
        pass


class _SpyEyes:
    """Models the eye device's pin lifecycle.

    The real bug (#1) was ``connect()`` calling ``stop()``, whose ``deinit()``
    killed the pins so every later write silently failed. Here ``stop()`` marks
    the device unusable and any subsequent cue raises — exactly what a write to a
    deinitialised pin does — so a ``connect()`` that stops the eyes is detectable.
    """

    def __init__(self):
        self.usable = True
        self.stopped = False
        self.authored = []          # note_authored() history (per-frame channel)
        self.cues = []              # expressive cue history (blink/double/hold)

    def _require_usable(self):
        if not self.usable:
            raise RuntimeError("eye pins deinitialised (stop() was called)")

    def note_authored(self, state):
        self._require_usable()
        self.authored.append(int(state))

    def blink(self, n=1):
        self._require_usable()
        self.cues.append(("blink", int(n)))

    def double_blink(self):
        self._require_usable()
        self.cues.append(("double_blink", None))

    def hold_open(self, seconds=1.0):
        self._require_usable()
        self.cues.append(("hold_open", float(seconds)))

    def stop(self):
        self.stopped = True
        self.usable = False


class _SpyImuUnavailable:
    """An IMU whose constructor fails, as when I2C is disabled on the Pi."""

    def __init__(self, sampling_freq=50):
        raise RuntimeError("No Hardware I2C on (scl,sda)=(3,2)")


class _SpySounds:
    def __init__(self, volume=1.0, sound_directory="./"):
        pass


class _SpyProjector:
    def __init__(self):
        self.on = False


@pytest.fixture
def fake_hw(monkeypatch):
    """Inject spy device modules so ``RealRobot.connect`` needs no hardware."""
    _SpyAntennas.instances = 0

    def _mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        monkeypatch.setitem(sys.modules, name, m)

    _mod("mini_bdx_runtime.rustypot_position_hwi", HWI=_SpyHWI)
    _mod("mini_bdx_runtime.raw_imu", Imu=_SpyImu)
    _mod("mini_bdx_runtime.antennas", Antennas=_SpyAntennas)
    _mod("mini_bdx_runtime.duck_config", DuckConfig=_SpyDuckConfig)
    _mod("mini_bdx_runtime.eyes", Eyes=_SpyEyes)
    _mod("mini_bdx_runtime.sounds", Sounds=_SpySounds)
    _mod("mini_bdx_runtime.projector", Projector=_SpyProjector)
    yield


def _make(**kwargs):
    from mini_bdx_runtime.anim.real_robot import RealRobot
    return RealRobot(**kwargs)


def test_construct_does_not_touch_antennas(fake_hw):
    robot = _make()
    assert robot.antennas is None
    assert _SpyAntennas.instances == 0


def test_connect_does_not_energise_antennas_or_bus(fake_hw):
    robot = _make()
    robot.connect()
    # The antenna PWM was never constructed just by connecting.
    assert robot.antennas is None
    assert _SpyAntennas.instances == 0
    # The bus was opened but no torque-producing call was made.
    assert robot.hwi is not None
    assert robot.hwi.actuated == []


def test_neutral_command_does_not_energise_antennas(fake_hw):
    """The BOOT/DISARMED/FAULT 'show off' path is set_antennas(0, 0)."""
    robot = _make()
    robot.connect()
    robot.shutdown_show()          # -> set_antennas(0.0, 0.0)
    robot.set_antennas(0.0, 0.0)
    assert robot.antennas is None
    assert _SpyAntennas.instances == 0


def test_first_real_command_energises_and_drives_antennas(fake_hw):
    robot = _make()
    robot.connect()
    robot.set_antennas(0.3, -0.4)
    assert _SpyAntennas.instances == 1
    assert robot.antennas is not None
    assert robot.antennas.left == pytest.approx(0.3)
    assert robot.antennas.right == pytest.approx(-0.4)
    # A subsequent command reuses the same device (no re-construction).
    robot.set_antennas(0.0, 0.0)
    assert _SpyAntennas.instances == 1
    assert robot.antennas.left == pytest.approx(0.0)


def test_connect_actuate_false_never_energises_antennas(fake_hw):
    robot = _make()
    robot.connect(actuate=False)
    robot.set_antennas(0.9, -0.9)  # even a real command stays dark
    assert robot.antennas is None
    assert _SpyAntennas.instances == 0


def test_enable_antennas_false_never_energises(fake_hw):
    robot = _make(enable_antennas=False)
    robot.connect()
    robot.set_antennas(0.9, -0.9)
    assert robot.antennas is None
    assert _SpyAntennas.instances == 0


# --- eyes: connect must leave the eye device USABLE (regression for bug #1) ---
def test_connect_leaves_eyes_usable_not_deinitialised(fake_hw):
    """connect() used to build Eyes() then immediately stop() it, which
    deinitialised the GPIO pins; every later set_eyes() then wrote to dead pins
    and the swallowed exception meant the eyes never lit. connect() must now
    leave the eye device alive so the per-frame eye channel reaches hardware."""
    robot = _make()
    robot.connect()
    assert robot.eyes is not None
    assert robot.eyes.usable is True          # pins NOT deinitialised by connect
    assert robot.eyes.stopped is False        # background blink thread not stopped
    # An authored eye channel (1->0 edge) actually reaches the live device.
    robot.set_eyes(1)
    robot.set_eyes(0)
    assert robot.eyes.authored == [1, 0]


def test_set_eye_event_maps_cues_to_hardware(fake_hw):
    """Expressive clip eye cues reach the device with the right mapping."""
    robot = _make()
    robot.connect()
    robot.set_eye_event("wide")
    robot.set_eye_event("happy")
    robot.set_eye_event("blink")
    assert robot.eyes.cues == [
        ("hold_open", 1.0),
        ("double_blink", None),
        ("blink", 1),
    ]


# --- IMU optional for dock/head, but tilt marked invalid (safety) -----------
def test_connect_tolerates_missing_imu(fake_hw, monkeypatch):
    """A robot with I2C disabled (no BNO055) must still connect for the dock /
    head-only path: connect() degrades to self.imu = None with a warning."""
    import types
    m = types.ModuleType("mini_bdx_runtime.raw_imu")
    m.Imu = _SpyImuUnavailable
    monkeypatch.setitem(sys.modules, "mini_bdx_runtime.raw_imu", m)
    robot = _make()
    robot.connect()
    assert robot.imu is None


def test_read_reports_tilt_invalid_without_imu(fake_hw, monkeypatch):
    """Without an IMU, read() falls back to zero tilt but MUST flag it invalid
    so the FSM cannot mistake the placeholder 0.0 for 'upright' and walk blind."""
    import types
    m = types.ModuleType("mini_bdx_runtime.raw_imu")
    m.Imu = _SpyImuUnavailable
    monkeypatch.setitem(sys.modules, "mini_bdx_runtime.raw_imu", m)
    robot = _make()
    robot.connect()

    class _PosHWI:
        def get_present_positions(self):
            return np.zeros(14)

        def get_present_velocities(self):
            return np.zeros(14)
    robot.hwi = _PosHWI()
    from mini_bdx_runtime.anim.hardware import OperatorInput
    snap = robot.read(OperatorInput())
    assert snap is not None
    assert snap.tilt_rad == 0.0
    assert snap.tilt_valid is False


def test_read_reports_tilt_valid_with_imu(fake_hw):
    """With a working IMU, read() reports tilt_valid=True."""
    robot = _make()
    robot.connect()

    class _PosHWI:
        def get_present_positions(self):
            return np.zeros(14)

        def get_present_velocities(self):
            return np.zeros(14)
    robot.hwi = _PosHWI()
    from mini_bdx_runtime.anim.hardware import OperatorInput
    snap = robot.read(OperatorInput())
    assert snap is not None
    assert snap.tilt_valid is True
