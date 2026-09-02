"""Mockable hardware abstraction for the animation runtime (plan §4.3, §6.5).

The mode FSM and the safety logic must be testable **without a robot** (plan
Phase 3/4: "hardware abstraction must be mockable"). The existing runtime talks
to hardware through a scattered set of concrete objects (``HWI``, ``Imu``,
``FeetContacts``, ``Antennas``, ``Eyes``, ``Sounds``, ``Projector``) that each
open a device on construction, so they cannot be imported off-robot. Rather than
restructure that code, we introduce a thin :class:`RobotInterface` seam:

* :class:`SensorSnapshot` — everything read **once** at the top of a control
  tick (plan §6.1 single-clock rule). The FSM and safety monitor consume only
  this, so they never touch a device.
* :class:`RobotInterface` — the write side (torque, joint targets, show
  functions) plus :meth:`RobotInterface.read` returning a snapshot.
* :class:`MockRobot` — a pure-Python fake used by the unit tests and by the
  MuJoCo validation harness; it records every command and lets a test script
  feed synthetic sensor values.

The real hardware adapter lives in :mod:`mini_bdx_runtime.anim.real_robot`, which
imports the on-robot device modules lazily so this module stays importable on a
laptop / CI with no ``board`` / ``rustypot`` present.

**Temperature / current.** ``rustypot==0.1.0`` (the pinned bus HWI) exposes only
``read_present_position`` / ``read_present_velocity``; it does **not** surface the
Feetech STS3215 *Present Temperature* (register 0x3F) or *Present Current*
(0x45). So :attr:`SensorSnapshot.temperatures_c` / ``currents_a`` are ``None`` on
the real robot today, and the :class:`~mini_bdx_runtime.anim.safety.ThermalManager`
degrades to a **time-based** duty limit (max continuous demo duration + cooldown)
plus the load-relieving dock posture. If a future ``rustypot`` adds the reads,
the real adapter fills these fields and the same ThermalManager enforces the
per-servo limits with no other change. See :mod:`.safety`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# 14-DOF Feetech bus order (Appendix A). Head/neck = indices 5..8; the rest are
# the 10 leg DOF. Kept local (not imported from open_duck_anim) so this module
# has no hard dependency on the core for a bare hardware mock.
N_DOFS = 14
HEAD_HW_SLICE = slice(5, 9)
LEG_HW_INDICES = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)


@dataclass
class OperatorInput:
    """Discrete operator / dock intents sampled each tick (plan §4.3).

    These are the *requests* that drive FSM transitions. They are deliberately
    separated from continuous sensor values so a test can drive the FSM through
    every transition by toggling booleans, with no robot.
    """

    # Deadman / e-stop asserted THIS tick (latched -> FAULT by the FSM/safety).
    estop: bool = False
    # Explicit dock confirmation (operator button or dock sense line). Required
    # to enter DOCK_DEMO (plan §4.3: "dockedness is never assumed").
    dock_confirmed: bool = False
    # Explicit off-dock confirmation (leaving the dock for STAND).
    offdock_confirmed: bool = False
    # Operator "arm" request: BOOT/DISARMED -> ARMING.
    arm_requested: bool = False
    # Explicit operator reset out of the latched FAULT (plan §4.3).
    fault_reset: bool = False
    # Locomotion command [vx, vy, wz] (drives STAND<->WALK; zero == stand).
    locomotion_command: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class SensorSnapshot:
    """All sensor values read once at the top of a control tick (plan §6.1).

    ``tilt_rad`` is supplied by the caller so the *source* of tilt is explicit
    and swappable: the MuJoCo harness passes the true base tilt; the on-robot
    adapter estimates it from the accelerometer gravity vector (documented in
    the real adapter). The FSM/safety code never has to know which.
    """

    t_monotonic: float
    joint_positions: np.ndarray             # (14,) rad, bus order
    joint_velocities: np.ndarray            # (14,) rad/s, bus order
    tilt_rad: float                         # roll/pitch magnitude from vertical
    feet_contacts: np.ndarray               # (2,) [left, right] in {0,1}
    # Whether ``tilt_rad`` reflects a real tilt measurement. False means the IMU
    # was unavailable and ``tilt_rad`` is a zero placeholder — it must NOT be
    # trusted as "upright". Balancing modes (STAND/WALK) require this True; the
    # non-balancing DOCK_DEMO / head-only path may run with it False.
    tilt_valid: bool = True
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accelero: np.ndarray = field(default_factory=lambda: np.zeros(3))
    temperatures_c: Optional[np.ndarray] = None   # (14,) or None if unsupported
    currents_a: Optional[np.ndarray] = None       # (14,) or None if unsupported
    operator: OperatorInput = field(default_factory=OperatorInput)

    def __post_init__(self):
        self.joint_positions = np.asarray(self.joint_positions, dtype=np.float64)
        self.joint_velocities = np.asarray(self.joint_velocities, dtype=np.float64)
        self.feet_contacts = np.asarray(self.feet_contacts, dtype=np.float64)
        if self.joint_positions.shape != (N_DOFS,):
            raise ValueError("joint_positions must be length 14")
        if self.joint_velocities.shape != (N_DOFS,):
            raise ValueError("joint_velocities must be length 14")


class RobotInterface(ABC):
    """The write side of the hardware seam plus a snapshot read.

    One thread owns this object exclusively (plan §6.1). All methods are called
    from the single control-loop thread.
    """

    @abstractmethod
    def read(self, operator: OperatorInput) -> Optional[SensorSnapshot]:
        """Read all sensors for this tick. Return ``None`` on a transient read
        failure (the caller skips the tick, as the current runtime does)."""

    @abstractmethod
    def set_gains(self, kps: np.ndarray, kds: np.ndarray) -> None:
        """Set per-joint position/velocity gains (14 each)."""

    @abstractmethod
    def set_joint_targets(self, targets: np.ndarray) -> None:
        """Command the 14-DOF bus targets (rad, bus order)."""

    @abstractmethod
    def torque_off(self) -> None:
        """Disable torque on all bus servos (FAULT: torque-off policy)."""

    @abstractmethod
    def set_antennas(self, left_norm: float, right_norm: float) -> None:
        """Command normalised [-1,1] antenna positions (plan §5.2 precedence:
        the runtime drives antennas only from show_functions)."""

    @abstractmethod
    def set_eyes(self, state: int) -> None:
        """Set discrete eye state (0/1)."""

    @abstractmethod
    def play_sound(self, name: str) -> None:
        """Fire a discrete sound event."""

    @abstractmethod
    def set_projector(self, on: bool) -> None:
        """Set the projector on/off."""

    def shutdown_show(self) -> None:
        """Neutralise/disable all show hardware (FAULT/e-stop): antennas neutral,
        sounds/projector off (plan §6.5-b). Default composes the primitives."""
        self.set_antennas(0.0, 0.0)
        self.set_projector(False)


class MockRobot(RobotInterface):
    """Pure-Python fake robot for tests and the MuJoCo harness.

    Holds a mutable ``next_snapshot`` (or a callable ``snapshot_fn``) that the
    caller sets each tick, and records every command in public attributes so a
    test can assert on exactly what the runtime did. No devices, no threads.
    """

    def __init__(self, snapshot_fn=None):
        self.snapshot_fn = snapshot_fn
        self.next_snapshot: Optional[SensorSnapshot] = None
        self.fail_read: bool = False

        # Recorded command history.
        self.kps: Optional[np.ndarray] = None
        self.kds: Optional[np.ndarray] = None
        self.last_targets: Optional[np.ndarray] = None
        self.target_history: List[np.ndarray] = []
        self.antenna_history: List[Tuple[float, float]] = []
        self.eye_history: List[int] = []
        self.eye_event_history: List[str] = []
        self.sound_history: List[str] = []
        self.projector_state: bool = False
        self.projector_history: List[bool] = []
        self.torque_is_off: bool = False
        self.torque_off_count: int = 0
        self.show_shutdown_count: int = 0

    def read(self, operator: OperatorInput) -> Optional[SensorSnapshot]:
        if self.fail_read:
            return None
        if self.snapshot_fn is not None:
            return self.snapshot_fn(operator)
        snap = self.next_snapshot
        if snap is not None:
            # Attach the operator input for this tick (tests usually build the
            # snapshot without it and pass operator separately).
            snap.operator = operator
        return snap

    def set_gains(self, kps: np.ndarray, kds: np.ndarray) -> None:
        self.kps = np.asarray(kps, dtype=np.float64).copy()
        self.kds = np.asarray(kds, dtype=np.float64).copy()

    def set_joint_targets(self, targets: np.ndarray) -> None:
        t = np.asarray(targets, dtype=np.float64).copy()
        if t.shape != (N_DOFS,):
            raise ValueError("targets must be length 14")
        self.torque_is_off = False
        self.last_targets = t
        self.target_history.append(t)

    def torque_off(self) -> None:
        self.torque_is_off = True
        self.torque_off_count += 1

    def set_antennas(self, left_norm: float, right_norm: float) -> None:
        self.antenna_history.append((float(left_norm), float(right_norm)))

    def set_eyes(self, state: int) -> None:
        self.eye_history.append(int(state))

    def set_eye_event(self, value: str) -> None:
        self.eye_event_history.append(str(value))

    def play_sound(self, name: str) -> None:
        self.sound_history.append(str(name))

    def set_projector(self, on: bool) -> None:
        self.projector_state = bool(on)
        self.projector_history.append(bool(on))

    def shutdown_show(self) -> None:
        self.show_shutdown_count += 1
        super().shutdown_show()
