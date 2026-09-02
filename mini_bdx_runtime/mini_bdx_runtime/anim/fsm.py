"""Mode FSM with safe startup, quantitative guards and a latched fault (§4.3).

States (plan §4.3):

    BOOT      BOOT/DISARMED: torque off, read encoders, no motion.
    ARMING    init targets to the MEASURED pose (never snap to init_pos), ramp
              torque and targets to nominal over T_arm.
    DOCK_DEMO policy bypassed, legs held (load-relieving dock posture), head/
              neck/antennas/eyes animated. Entry requires explicit dock confirm.
    STAND     locomotion policy, zero command + head injection.
    WALK      same locomotion policy, vx/vy/wz command + head injection.
    FAULT     latched: requires an explicit operator reset back to BOOT.

The FSM is **pure and fully mockable**: :meth:`ModeFSM.update` consumes only a
:class:`~mini_bdx_runtime.anim.hardware.SensorSnapshot`, the previous tick's
commanded 14-DOF targets, and any externally-detected fault (watchdog/thermal
from :mod:`.safety`). It never touches a device, so every transition, guard
rejection, the arming ramp and the latched fault are unit-testable with no robot.

Transition guards are **quantitative** (plan §4.3), not "near safe pose":

    | metric                         | threshold (tune on hardware) |
    | max joint position error       | <= 0.05 rad                  |
    | max joint velocity             | <= 0.5 rad/s                 |
    | IMU tilt (roll/pitch)          | <= 0.10 rad                  |
    | foot contact (both for STAND)  | as required by target mode   |
    | dwell                          | hold >= 0.3 s before firing  |
    | hysteresis                     | leave band by >= 20% to re-arm |

Tilt during OPERATIONAL modes uses a looser bound (``operational_tilt_rad``,
default 0.15 rad = the plan's Phase-4 acceptance bound) so normal walking sway
does not fault; the tight 0.10 rad guard governs the ARMING->operational
handoff, when the robot should be quiescent.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from .hardware import SensorSnapshot, OperatorInput, N_DOFS


class FSMState(Enum):
    BOOT = "boot"          # BOOT / DISARMED
    ARMING = "arming"
    DOCK_DEMO = "dock_demo"
    STAND = "stand"
    WALK = "walk"
    FAULT = "fault"        # latched


# Modes in which the animation engine is evaluated as a balancing/operational
# mode (head injection + show). DOCK_DEMO is operational but non-balancing.
_OPERATIONAL = (FSMState.DOCK_DEMO, FSMState.STAND, FSMState.WALK)


@dataclass
class FSMConfig:
    """Quantitative guard thresholds and timings (plan §4.3; tune on hardware)."""

    # --- guards G (ARMING -> operational, and dock<->stand) ---
    max_pos_error_rad: float = 0.05
    max_joint_vel_rad_s: float = 0.5
    arming_tilt_rad: float = 0.10          # tight bound at the quiescent handoff
    operational_tilt_rad: float = 0.15     # Phase-4 acceptance tilt bound
    dwell_s: float = 0.3                   # guard must hold this long
    hysteresis_frac: float = 0.20          # re-entry margin (20%)

    # --- arming ramp ---
    t_arm_s: float = 1.5                   # ramp duration (proposed 1.0-2.0 s)
    arm_kp_low_frac: float = 0.10          # torque fraction at ramp start
    arm_timeout_s: float = 4.0             # ARMING must complete within this
    # If measured deviates from the arming reference by more than this at any
    # point during the ramp, something is wrong (a servo stalled / bus fault) ->
    # FAULT (plan §4.3: "a target/measured mismatch beyond tolerance").
    arm_track_tol_rad: float = 0.35

    # --- locomotion command deadband (STAND<->WALK) ---
    walk_cmd_deadband: float = 0.02
    # WALK requires both feet? No — walking is single/double support; only STAND
    # and the dock handoff require both-feet contact.
    require_both_feet_for_stand: bool = True


@dataclass
class GuardBand:
    """A one-sided ``metric <= threshold`` guard with 20% hysteresis (plan §4.3).

    ``inside`` latches: once the metric exceeds ``threshold`` the band opens and
    the metric must fall below ``threshold*(1-hysteresis)`` to close again. This
    prevents chatter around the boundary. State is explicit so the FSM can reset
    it on mode changes.
    """

    threshold: float
    hysteresis_frac: float = 0.20
    inside: bool = True

    def update(self, metric: float) -> bool:
        lo = self.threshold * (1.0 - self.hysteresis_frac)
        if self.inside:
            if metric > self.threshold:
                self.inside = False
        else:
            if metric <= lo:
                self.inside = True
        return self.inside

    def reset(self, inside: bool = False) -> None:
        self.inside = inside


@dataclass
class FSMStatus:
    """Per-tick FSM output for the controller and for logging/tests."""

    state: FSMState
    prev_state: FSMState
    changed: bool
    # Arming reference target (14,) and kp fraction, valid only while ARMING.
    arming_reference: Optional[np.ndarray] = None
    arming_kp_frac: float = 1.0
    arming_progress: float = 1.0
    # Which discrete guard metrics currently hold (for diagnostics/tests).
    guards_hold: bool = False
    fault_reason: str = ""


class ModeFSM:
    """The mode finite-state machine (plan §4.3).

    Usage per tick::

        status = fsm.update(snapshot, commanded_targets, external_fault, reason)

    ``external_fault`` lets :mod:`.safety` (watchdogs, thermal, e-stop) force the
    latched FAULT while keeping all the *transition* logic here.
    """

    def __init__(self, init_pos_14: np.ndarray, config: Optional[FSMConfig] = None,
                 self_check=None):
        self.cfg = config or FSMConfig()
        self.init_pos = np.asarray(init_pos_14, dtype=np.float64).copy()
        if self.init_pos.shape != (N_DOFS,):
            raise ValueError("init_pos_14 must be length 14")
        # Optional callable(snapshot) -> (ok: bool, reason: str) run at BOOT.
        self._self_check = self_check

        self.state = FSMState.BOOT
        self._fault_reason = ""

        # Guard bands (recreated on transitions that need a fresh dwell).
        self._pos_band = GuardBand(self.cfg.max_pos_error_rad, self.cfg.hysteresis_frac)
        self._vel_band = GuardBand(self.cfg.max_joint_vel_rad_s, self.cfg.hysteresis_frac)
        self._tilt_band = GuardBand(self.cfg.arming_tilt_rad, self.cfg.hysteresis_frac)
        self._dwell_accum = 0.0

        # Arming state.
        self._arm_measured0: Optional[np.ndarray] = None
        self._arm_t0: float = 0.0
        self._last_t: Optional[float] = None

    # --- public API -----------------------------------------------------------
    @property
    def fault_reason(self) -> str:
        return self._fault_reason

    def to_fault(self, reason: str) -> None:
        """Force the latched FAULT (used by safety monitor and internally)."""
        if self.state != FSMState.FAULT:
            self._fault_reason = reason
        self.state = FSMState.FAULT

    def engine_mode(self) -> Optional[str]:
        """Map the FSM state to an ``open_duck_anim`` engine mode string, or None
        when the engine must not be evaluated (BOOT/ARMING/FAULT)."""
        if self.state == FSMState.DOCK_DEMO:
            return "dock"
        if self.state == FSMState.STAND:
            return "stand"
        if self.state == FSMState.WALK:
            return "walk"
        return None

    def update(
        self,
        snap: SensorSnapshot,
        commanded_targets: Optional[np.ndarray],
        external_fault: bool = False,
        external_fault_reason: str = "",
    ) -> FSMStatus:
        prev = self.state
        dt = self._tick_dt(snap.t_monotonic)

        # E-stop and any externally-detected fault latch immediately, from ANY
        # state (plan §4.3: deadman/watchdog/thermal -> FAULT).
        if self.state != FSMState.FAULT:
            if snap.operator.estop:
                self.to_fault("e-stop asserted")
            elif external_fault:
                self.to_fault(external_fault_reason or "external fault")

        if self.state != FSMState.FAULT:
            self._step(snap, commanded_targets, dt)

        status = self._build_status(prev)
        return status

    # --- internal -------------------------------------------------------------
    def _tick_dt(self, t: float) -> float:
        if self._last_t is None or t <= self._last_t:
            dt = 0.0
        else:
            dt = t - self._last_t
        self._last_t = t
        return dt

    def _step(self, snap: SensorSnapshot, commanded_targets, dt: float) -> None:
        st = self.state
        op = snap.operator
        if st == FSMState.BOOT:
            self._step_boot(snap, op)
        elif st == FSMState.ARMING:
            self._step_arming(snap, commanded_targets, dt)
        elif st == FSMState.STAND:
            self._step_stand(snap, commanded_targets, dt, op)
        elif st == FSMState.WALK:
            self._step_walk(snap, op)
        elif st == FSMState.DOCK_DEMO:
            self._step_dock(snap, commanded_targets, dt, op)

    def _step_boot(self, snap, op: OperatorInput) -> None:
        if op.fault_reset:
            return  # already disarmed; nothing to reset
        if op.arm_requested:
            if self._self_check is not None:
                ok, reason = self._self_check(snap)
                if not ok:
                    self.to_fault("self-check failed: %s" % reason)
                    return
            self._enter_arming(snap)

    def _enter_arming(self, snap) -> None:
        self.state = FSMState.ARMING
        self._arm_measured0 = snap.joint_positions.copy()
        self._arm_t0 = snap.t_monotonic
        self._reset_guards(inside=False)

    def _arming_progress(self, t: float) -> float:
        if self.cfg.t_arm_s <= 0:
            return 1.0
        return float(np.clip((t - self._arm_t0) / self.cfg.t_arm_s, 0.0, 1.0))

    def arming_reference(self, t: float) -> np.ndarray:
        """Ramped reference target during ARMING: lerp(measured0 -> init_pos)."""
        s = self._arming_progress(t)
        assert self._arm_measured0 is not None
        return (1.0 - s) * self._arm_measured0 + s * self.init_pos

    def arming_kp_frac(self, t: float) -> float:
        s = self._arming_progress(t)
        lo = self.cfg.arm_kp_low_frac
        return lo + (1.0 - lo) * s

    def _step_arming(self, snap, commanded_targets, dt: float) -> None:
        t = snap.t_monotonic
        # Tracking-mismatch guard: measured must follow the ramp reference.
        ref = self.arming_reference(t)
        track_err = float(np.max(np.abs(snap.joint_positions - ref)))
        if track_err > self.cfg.arm_track_tol_rad:
            self.to_fault("ARMING tracking mismatch %.3f rad > %.3f"
                          % (track_err, self.cfg.arm_track_tol_rad))
            return
        # Timeout.
        if (t - self._arm_t0) > self.cfg.arm_timeout_s:
            self.to_fault("ARMING timeout (%.1f s)" % self.cfg.arm_timeout_s)
            return
        # Ramp not finished yet -> keep arming.
        if self._arming_progress(t) < 1.0:
            self._dwell_accum = 0.0
            return
        # Ramp complete: evaluate guards G against the NOMINAL target.
        hold = self._guards_hold(snap, self.init_pos, dt,
                                 tilt_thr=self.cfg.arming_tilt_rad,
                                 require_both_feet=True)
        if not hold:
            return
        # Guards held for the dwell -> choose destination from confirmation.
        op = snap.operator
        if op.dock_confirmed:
            self._enter_operational(FSMState.DOCK_DEMO)
        elif op.offdock_confirmed:
            self._enter_operational(FSMState.STAND)
        # else: hold in ARMING (armed, waiting for a destination confirmation).

    def _enter_operational(self, state: FSMState) -> None:
        self.state = state
        self._reset_guards(inside=False)

    def _step_stand(self, snap, commanded_targets, dt: float, op: OperatorInput) -> None:
        # Operational tilt fault (looser bound).
        if snap.tilt_rad > self.cfg.operational_tilt_rad:
            self.to_fault("tilt %.3f rad > operational bound %.3f"
                          % (snap.tilt_rad, self.cfg.operational_tilt_rad))
            return
        # STAND -> WALK on a nonzero locomotion command (no policy swap).
        if self._cmd_active(op.locomotion_command):
            self._enter_operational(FSMState.WALK)
            return
        # STAND -> DOCK_DEMO on dock confirm AND guards G.
        if op.dock_confirmed:
            if self._guards_hold(snap, commanded_targets, dt,
                                 tilt_thr=self.cfg.arming_tilt_rad,
                                 require_both_feet=True):
                self._enter_operational(FSMState.DOCK_DEMO)

    def _step_walk(self, snap, op: OperatorInput) -> None:
        if snap.tilt_rad > self.cfg.operational_tilt_rad:
            self.to_fault("tilt %.3f rad > operational bound %.3f"
                          % (snap.tilt_rad, self.cfg.operational_tilt_rad))
            return
        # WALK -> STAND when the command returns to zero (no policy swap).
        if not self._cmd_active(op.locomotion_command):
            self._enter_operational(FSMState.STAND)

    def _step_dock(self, snap, commanded_targets, dt: float, op: OperatorInput) -> None:
        # DOCK_DEMO -> STAND on off-dock confirm AND guards G.
        if op.offdock_confirmed:
            if self._guards_hold(snap, commanded_targets, dt,
                                 tilt_thr=self.cfg.arming_tilt_rad,
                                 require_both_feet=True):
                self._enter_operational(FSMState.STAND)

    # --- guard evaluation -----------------------------------------------------
    def _cmd_active(self, cmd: Tuple[float, float, float]) -> bool:
        return float(np.max(np.abs(np.asarray(cmd, dtype=np.float64)))) > self.cfg.walk_cmd_deadband

    def _guards_hold(self, snap, target, dt: float, tilt_thr: float,
                     require_both_feet: bool) -> bool:
        """Evaluate the full guard set with hysteresis + dwell (plan §4.3)."""
        # Position error vs the current commanded target.
        if target is None:
            pos_err = np.inf
        else:
            target = np.asarray(target, dtype=np.float64)
            pos_err = float(np.max(np.abs(snap.joint_positions - target)))
        vel = float(np.max(np.abs(snap.joint_velocities)))
        tilt = float(snap.tilt_rad)

        self._tilt_band.threshold = tilt_thr
        pos_ok = self._pos_band.update(pos_err)
        vel_ok = self._vel_band.update(vel)
        tilt_ok = self._tilt_band.update(tilt)
        feet_ok = True
        if require_both_feet and self.cfg.require_both_feet_for_stand:
            feet_ok = bool(np.all(snap.feet_contacts >= 0.5))

        all_ok = pos_ok and vel_ok and tilt_ok and feet_ok
        if all_ok:
            self._dwell_accum += dt
        else:
            self._dwell_accum = 0.0
        return all_ok and self._dwell_accum >= self.cfg.dwell_s

    def _reset_guards(self, inside: bool) -> None:
        self._pos_band.reset(inside)
        self._vel_band.reset(inside)
        self._tilt_band.reset(inside)
        self._dwell_accum = 0.0

    def _build_status(self, prev: FSMState) -> FSMStatus:
        arming_ref = None
        arm_kp = 1.0
        arm_prog = 1.0
        if self.state == FSMState.ARMING and self._arm_measured0 is not None and self._last_t is not None:
            arming_ref = self.arming_reference(self._last_t)
            arm_kp = self.arming_kp_frac(self._last_t)
            arm_prog = self._arming_progress(self._last_t)
        return FSMStatus(
            state=self.state,
            prev_state=prev,
            changed=(self.state != prev),
            arming_reference=arming_ref,
            arming_kp_frac=arm_kp,
            arming_progress=arm_prog,
            guards_hold=(self._dwell_accum >= self.cfg.dwell_s),
            fault_reason=self._fault_reason,
        )

    def request_reset(self, snap: SensorSnapshot) -> bool:
        """Explicit operator reset FAULT -> BOOT (plan §4.3: never auto-cleared).

        Returns True if a reset happened. The caller (controller) invokes this
        only when it is safe to re-disarm (torque already commanded off)."""
        if self.state == FSMState.FAULT and snap.operator.fault_reset:
            self.state = FSMState.BOOT
            self._fault_reason = ""
            self._arm_measured0 = None
            self._reset_guards(inside=False)
            return True
        return False
