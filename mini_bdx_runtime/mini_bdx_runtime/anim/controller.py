"""Animation controller — the single integration hub (plan §6.1, §6.2, §6.5).

Ties together the four pieces that already exist and must NOT be reimplemented:

* ``open_duck_anim.Engine`` — the three-layer blend engine (head offsets + show).
* :class:`~mini_bdx_runtime.anim.fsm.ModeFSM` — the mode/safety state machine.
* :class:`~mini_bdx_runtime.anim.safety.SafetyMonitor` — watchdogs + thermal +
  e-stop + the fault-action decision.
* :class:`~mini_bdx_runtime.anim.hardware.RobotInterface` — the mockable IO seam.

Design rules enforced here (all from the plan):

* **Evaluate the engine exactly once per control tick**, from the single
  monotonic timestamp already carried on the :class:`SensorSnapshot`. No second
  timer thread (plan §6.1: a separate animation clock drifts against the policy
  clock).
* **Route head output through the existing additive path.** In STAND/WALK the
  engine emits a *relative* head offset; the controller returns it so the main
  loop writes it into ``last_commands[3:7]`` (so it also enters the obs) and the
  runtime's existing line ``head_motor_targets = last_commands[3:] +
  motor_targets[5:9]`` actuates the head. We do NOT remove that line.
* **Head output already passes the safety envelope** — the Engine enforces
  ``DEFAULT_ENVELOPE`` by default; we build it with ``.derated()`` (×0.5) for
  first hardware trials, exposed as the clearly-labelled
  :attr:`ControllerConfig.envelope_derating` flag.
* **Capability matrix (plan §6.2).** STAND/WALK: legs are policy-owned, so any
  animated leg channel is discarded (the Engine already returns ``leg_targets
  is None`` there; we assert it and never write legs from animation). DOCK_DEMO:
  legs are *held* at ``init_pos`` (the Engine's ``leg_targets``) and the head is
  driven by absolute ``head_targets``, bypassing the policy entirely.
* **Final bus safety.** After mode selection, the FINAL 14-DOF bus targets pass
  a :class:`JointLimiter` (MJCF ``jnt_range``) and the re-enabled
  :class:`JointRateLimiter` (``max_motor_velocity = 5.24 rad/s``) — on the bus
  targets, not on the animation command (plan §6.4).

Usage — two-phase so the RL loop can inject the head offset into the obs before
running the policy, while still evaluating the engine only once::

    plan = controller.prepare(operator, triggers=triggers)   # engine eval here
    if plan.needs_policy:                                     # STAND / WALK
        last_commands[3:7] = plan.head_offsets                # -> obs
        action = policy.infer(get_obs())
        policy_targets = init_pos + action * ACTION_SCALE
        out = controller.finalize(policy_targets)
    else:                                                     # DOCK/ARMING/...
        out = controller.finalize(None)

For the no-RL dock demo, :meth:`step` wraps both phases.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import _ensure_open_duck_anim

_ensure_open_duck_anim()
from open_duck_anim import (  # noqa: E402  (after sys.path bootstrap)
    Engine,
    Triggers,
    DEFAULT_ENVELOPE,
    JointLimiter,
    JointRateLimiter,
    AntennaSlewLimiter,
    MAX_MOTOR_VELOCITY,
    INIT_POS_14,
)

from .hardware import RobotInterface, OperatorInput, SensorSnapshot, N_DOFS
from .fsm import ModeFSM, FSMConfig, FSMState, FSMStatus
from .safety import SafetyMonitor, SafetyConfig, SafetyStatus, FaultAction


# Head joints occupy bus indices 5..8; legs are the remaining ten.
HEAD_SLICE = slice(5, 9)
LEG_HW_INDICES = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13)

# MJCF ``jnt_range`` for the 14-DOF bus order (scene_flat_terrain.xml, verified
# against the actuator/joint order in rustypot_position_hwi.py). Used as the
# final position clamp. Override via ``ControllerConfig.joint_limits``.
_MJCF_JOINT_LOW = np.array([
    -0.5236, -0.4363, -1.2217, -1.5708, -1.5708,   # left leg
    -0.3491, -0.7854, -2.7925, -0.5236,            # head (neck,pitch,yaw,roll)
    -0.5236, -0.4363, -0.5236, -1.5708, -1.5708,   # right leg
], dtype=np.float64)
_MJCF_JOINT_HIGH = np.array([
    0.5236, 0.4363, 0.5236, 1.5708, 1.5708,
    1.1345, 0.7854, 2.7925, 0.5236,
    0.5236, 0.4363, 1.2217, 1.5708, 1.5708,
], dtype=np.float64)

# Physical head joint limits (bus 5..8) handed to the Engine for DOCK absolute
# head targets — the real MJCF numbers, tighter/safer than a guess.
_HEAD_JOINT_LOW = _MJCF_JOINT_LOW[HEAD_SLICE].copy()
_HEAD_JOINT_HIGH = _MJCF_JOINT_HIGH[HEAD_SLICE].copy()


def make_head_engine(envelope_derating: Optional[float] = 0.5, background_clip=None):
    """Build an ``open_duck_anim.Engine`` configured for this robot.

    Single source of truth for the head safety envelope and physical head joint
    limits, shared by :class:`AnimationController` and by the RL walk loop's
    lightweight head-overlay path so both use identical safety numbers.

    ``envelope_derating`` scales the measured head envelope for early hardware
    trials (``0.5`` = plan §6.5 default). Pass ``None`` for the full
    ``DEFAULT_ENVELOPE``. The Engine ALWAYS enforces the envelope; there is no
    unbounded path here.
    """
    if envelope_derating is None:
        envelope = DEFAULT_ENVELOPE
    else:
        envelope = DEFAULT_ENVELOPE.derated(envelope_derating)
    return Engine(
        background=background_clip,
        head_joint_limits=(_HEAD_JOINT_LOW, _HEAD_JOINT_HIGH),
        head_envelope=envelope,
    )


@dataclass
class ControllerConfig:
    ctrl_dt: float = 0.02                       # 50 Hz

    # --- envelope derating (plan §6.5, Phase 4) ---
    # SAFETY: for first hardware trials the additive head envelope is derated to
    # 50% (``open_duck_anim`` ``HARDWARE_DERATING``). Relax toward 1.0 only as
    # hardware data accrues. Set to None to use the full DEFAULT_ENVELOPE.
    envelope_derating: Optional[float] = 0.5

    # --- gains (from v2_rl_walk_mujoco.py: leg kp=30, head kp=8, all kd=0) ---
    leg_kp: float = 30.0
    head_kp: float = 8.0
    kd: float = 0.0

    # --- final bus safety ---
    max_motor_velocity: float = MAX_MOTOR_VELOCITY   # 5.24 rad/s, re-enabled
    enable_rate_limit: bool = True
    enable_joint_clamp: bool = True

    # --- thermal duty model: which states count as "under load" ---
    # STAND/WALK actively balance under torque; DOCK holds init_pos (load
    # relief); BOOT/torque-off draw nothing. ARMING ramps torque -> counts.
    dock_is_load_relief: bool = True


@dataclass
class ControllerOutput:
    """Everything the caller/logging needs from one tick."""

    state: FSMState
    fsm: FSMStatus
    safety: SafetyStatus
    bus_targets: Optional[np.ndarray]     # 14-DOF commanded targets (None if torque off)
    head_offsets: np.ndarray              # (4,) additive head offset used this tick
    torque_off: bool
    engine_evaluated: bool
    antennas: Optional[tuple] = None      # (l,r) normalised, if driven this tick
    events_fired: List = field(default_factory=list)
    request_cooldown: bool = False


@dataclass
class _TickPlan:
    snapshot: SensorSnapshot
    fsm: FSMStatus
    safety: SafetyStatus
    engine_mode: Optional[str]
    engine_out: object
    head_offsets: np.ndarray
    needs_policy: bool
    under_load: bool


class AnimationController:
    def __init__(
        self,
        robot: RobotInterface,
        background_clip=None,
        config: Optional[ControllerConfig] = None,
        fsm_config: Optional[FSMConfig] = None,
        safety_config: Optional[SafetyConfig] = None,
        init_pos_14: Optional[np.ndarray] = None,
        joint_limits: Optional[tuple] = None,
    ):
        self.robot = robot
        self.cfg = config or ControllerConfig()
        self.init_pos = np.asarray(
            INIT_POS_14 if init_pos_14 is None else init_pos_14, dtype=np.float64
        ).copy()
        if self.init_pos.shape != (N_DOFS,):
            raise ValueError("init_pos_14 must be length 14")

        # Head safety envelope: derated for first hardware trials (plan §6.5).
        self.engine = make_head_engine(
            envelope_derating=self.cfg.envelope_derating,
            background_clip=background_clip,
        )

        self.fsm = ModeFSM(self.init_pos, fsm_config)
        self.safety = SafetyMonitor(safety_config)

        if joint_limits is None:
            low, high = _MJCF_JOINT_LOW, _MJCF_JOINT_HIGH
        else:
            low, high = joint_limits
        self.joint_limiter = JointLimiter(low, high)
        self.rate_limiter = JointRateLimiter(self.cfg.max_motor_velocity)
        self.antenna_slew = AntennaSlewLimiter()

        # Base gains (14,) in bus order.
        self._base_kp = np.empty(N_DOFS, dtype=np.float64)
        self._base_kp[list(LEG_HW_INDICES)] = self.cfg.leg_kp
        self._base_kp[HEAD_SLICE] = self.cfg.head_kp
        self._base_kd = np.full(N_DOFS, self.cfg.kd, dtype=np.float64)
        self._last_kp: Optional[np.ndarray] = None

        # Rate-limit / slew reference state.
        self._prev_bus_targets: Optional[np.ndarray] = None
        self._prev_antennas = np.zeros(2, dtype=np.float64)

        # Deadline watchdog bookkeeping.
        self._last_tick_wall: Optional[float] = None
        self._pending: Optional[_TickPlan] = None

    # -- helpers ---------------------------------------------------------------
    def heartbeat(self, t: float) -> None:
        """Register a fresh controller command (stale-command watchdog)."""
        self.safety.heartbeat(t)

    def _set_gains(self, kp: np.ndarray) -> None:
        if self._last_kp is None or not np.array_equal(kp, self._last_kp):
            self.robot.set_gains(kp, self._base_kd)
            self._last_kp = kp.copy()

    def _under_load(self, state: FSMState) -> bool:
        if state in (FSMState.STAND, FSMState.WALK, FSMState.ARMING):
            return True
        if state == FSMState.DOCK_DEMO:
            return not self.cfg.dock_is_load_relief
        return False

    # -- phase 1: read + FSM + safety + engine (single eval) -------------------
    def prepare(
        self,
        operator: OperatorInput,
        snapshot: Optional[SensorSnapshot] = None,
        triggers: Optional[Triggers] = None,
    ) -> Optional[_TickPlan]:
        """Read sensors, step safety + FSM, and evaluate the engine ONCE.

        Returns ``None`` on a transient sensor read failure (caller skips tick,
        matching the current runtime). Otherwise returns a plan for
        :meth:`finalize`.
        """
        if snapshot is None:
            snapshot = self.robot.read(operator)
            if snapshot is None:
                return None
        t = snapshot.t_monotonic

        # --- safety monitor (deadline uses the previous tick's wall duration) ---
        last_tick_dur = None
        if self._last_tick_wall is not None:
            last_tick_dur = max(0.0, t - self._last_tick_wall)
        self._last_tick_wall = t

        balancing = self.fsm.state in (FSMState.STAND, FSMState.WALK)
        under_load = self._under_load(self.fsm.state)
        safety = self.safety.update(
            snapshot, balancing=balancing, under_load=under_load,
            dt=self.cfg.ctrl_dt, last_tick_duration_s=last_tick_dur,
        )

        # --- FSM (latches on e-stop / external fault) ---
        fsm = self.fsm.update(
            snapshot, self._prev_bus_targets,
            external_fault=safety.fault, external_fault_reason=safety.fault_reason,
        )

        # --- engine: evaluate EXACTLY ONCE, only in operational modes ---
        engine_mode = self.fsm.engine_mode()
        head_offsets = np.zeros(4, dtype=np.float64)
        engine_out = None
        if engine_mode is not None:
            trg = triggers or Triggers()
            # Thermal cooldown -> controlled abort: cancel active clips so head
            # and show blend back to the owning controller over T_alpha/T_beta.
            if safety.request_cooldown:
                trg = Triggers(clips=list(trg.clips), joystick_offset=trg.joystick_offset,
                               cancel=True)
            engine_out = self.engine.evaluate(t, engine_mode, trg)
            head_offsets = np.asarray(engine_out.head_command_offsets, dtype=np.float64)

        needs_policy = self.fsm.state in (FSMState.STAND, FSMState.WALK)
        plan = _TickPlan(
            snapshot=snapshot, fsm=fsm, safety=safety, engine_mode=engine_mode,
            engine_out=engine_out, head_offsets=head_offsets,
            needs_policy=needs_policy, under_load=under_load,
        )
        self._pending = plan
        return plan

    # -- phase 2: mode selection + final bus safety + command ------------------
    def finalize(self, policy_targets_14: Optional[np.ndarray] = None) -> ControllerOutput:
        plan = self._pending
        if plan is None:
            raise RuntimeError("finalize() called before prepare()")
        self._pending = None
        snap = plan.snapshot
        state = self.fsm.state

        out = ControllerOutput(
            state=state, fsm=plan.fsm, safety=plan.safety,
            bus_targets=None, head_offsets=plan.head_offsets, torque_off=False,
            engine_evaluated=(plan.engine_out is not None),
            request_cooldown=plan.safety.request_cooldown,
        )

        # ---- FAULT (latched) ----
        if state == FSMState.FAULT:
            self._handle_fault(snap, plan.safety, out)
            return out

        # ---- BOOT / DISARMED: torque off, no motion ----
        if state == FSMState.BOOT:
            self.robot.torque_off()
            self.robot.shutdown_show()
            out.torque_off = True
            self._prev_bus_targets = None
            self._last_kp = None
            return out

        # ---- build the 14-DOF bus targets by mode ----
        if state == FSMState.ARMING:
            targets = self._targets_arming(plan)
        elif state == FSMState.DOCK_DEMO:
            targets = self._targets_dock(plan)
        elif state in (FSMState.STAND, FSMState.WALK):
            targets = self._targets_balancing(plan, policy_targets_14)
        else:  # pragma: no cover - defensive
            raise RuntimeError("unhandled state %r" % state)

        # ---- final bus safety: clamp position, then velocity, on FINAL targets ----
        targets = self._apply_final_safety(targets)

        self.robot.set_joint_targets(targets)
        self._prev_bus_targets = targets.copy()
        self.heartbeat(snap.t_monotonic)
        out.bus_targets = targets

        # ---- show functions (operational modes only) ----
        if state in (FSMState.DOCK_DEMO, FSMState.STAND, FSMState.WALK) and plan.engine_out is not None:
            self._drive_show(plan.engine_out, out)
        return out

    def step(
        self,
        operator: OperatorInput,
        snapshot: Optional[SensorSnapshot] = None,
        triggers: Optional[Triggers] = None,
    ) -> Optional[ControllerOutput]:
        """Convenience single-call tick for the no-RL dock demo."""
        plan = self.prepare(operator, snapshot=snapshot, triggers=triggers)
        if plan is None:
            return None
        if plan.needs_policy:
            raise RuntimeError(
                "step() used in a policy mode (%s); use prepare()/finalize()"
                % self.fsm.state
            )
        return self.finalize(None)

    # -- per-mode target construction -----------------------------------------
    def _targets_arming(self, plan: _TickPlan) -> np.ndarray:
        ref = plan.fsm.arming_reference
        if ref is None:
            ref = self.init_pos
        kp = self._base_kp * float(plan.fsm.arming_kp_frac)
        self._set_gains(kp)
        # Seed the rate-limit reference from the MEASURED pose on entry so the
        # first commanded step is bounded from where the robot actually is.
        if self._prev_bus_targets is None:
            self._prev_bus_targets = plan.snapshot.joint_positions.copy()
        return np.asarray(ref, dtype=np.float64).copy()

    def _targets_dock(self, plan: _TickPlan) -> np.ndarray:
        self._set_gains(self._base_kp)
        eo = plan.engine_out
        targets = self.init_pos.copy()
        # Legs held at init_pos (Engine leg_targets) — bypass policy entirely.
        if eo is not None and eo.leg_targets is not None:
            targets[list(LEG_HW_INDICES)] = np.asarray(eo.leg_targets, dtype=np.float64)
        # Head driven by ABSOLUTE joint targets (already clamped to head jnt
        # limits by the Engine in dock mode).
        if eo is not None and eo.head_targets is not None:
            targets[HEAD_SLICE] = np.asarray(eo.head_targets, dtype=np.float64)
        if self._prev_bus_targets is None:
            self._prev_bus_targets = plan.snapshot.joint_positions.copy()
        return targets

    def _targets_balancing(self, plan: _TickPlan, policy_targets_14) -> np.ndarray:
        if policy_targets_14 is None:
            raise RuntimeError(
                "STAND/WALK requires policy_targets_14; pass the policy output "
                "to finalize()"
            )
        self._set_gains(self._base_kp)
        targets = np.asarray(policy_targets_14, dtype=np.float64).copy()
        if targets.shape != (N_DOFS,):
            raise ValueError("policy_targets_14 must be length 14")
        # CAPABILITY MATRIX (plan §6.2): legs are policy-owned in STAND/WALK, so
        # any animated leg channel is discarded. The Engine already returns
        # leg_targets=None here; assert it so a future regression is caught.
        eo = plan.engine_out
        if eo is not None and eo.leg_targets is not None:  # pragma: no cover
            raise AssertionError("engine emitted leg targets in a balancing mode")
        # ADDITIVE HEAD PATH (v2_rl_walk_mujoco.py:310-311, kept): head offset is
        # added onto the policy's head targets. The main loop also writes the
        # same offset into last_commands[3:7] so it enters the obs.
        targets[HEAD_SLICE] = targets[HEAD_SLICE] + plan.head_offsets
        if self._prev_bus_targets is None:
            self._prev_bus_targets = plan.snapshot.joint_positions.copy()
        return targets

    def _apply_final_safety(self, targets: np.ndarray) -> np.ndarray:
        if self.cfg.enable_joint_clamp:
            targets = self.joint_limiter.clamp(targets)
        if self.cfg.enable_rate_limit and self._prev_bus_targets is not None:
            targets = self.rate_limiter.limit(
                self._prev_bus_targets, targets, self.cfg.ctrl_dt
            )
        return targets

    # -- fault handling --------------------------------------------------------
    def _handle_fault(self, snap: SensorSnapshot, safety: SafetyStatus,
                      out: ControllerOutput) -> None:
        # Neutralise show hardware first (antennas neutral, sounds/projector off).
        self.robot.shutdown_show()
        action = safety.fault_action
        if action == FaultAction.TORQUE_OFF:
            self.robot.torque_off()
            out.torque_off = True
            self._prev_bus_targets = None
            self._last_kp = None
        else:  # CONTROLLED_HOLD: keep last commanded pose energised.
            hold = self._prev_bus_targets
            if hold is None:
                hold = snap.joint_positions.copy()
            self._set_gains(self._base_kp)
            self.robot.set_joint_targets(hold)
            out.bus_targets = hold
            self._prev_bus_targets = hold.copy()
        # Explicit operator reset: FAULT -> BOOT (never auto-cleared). Only after
        # torque is off / hold is safe.
        self.fsm.request_reset(snap)

    # -- show functions --------------------------------------------------------
    def _drive_show(self, engine_out, out: ControllerOutput) -> None:
        show = engine_out.show
        # Antennas: slew-limit the normalised [-1,1] track, then command.
        tgt = np.array([show.antenna_l, show.antenna_r], dtype=np.float64)
        slewed = self.antenna_slew.limit(self._prev_antennas, tgt, self.cfg.ctrl_dt)
        self._prev_antennas = slewed
        self.robot.set_antennas(float(slewed[0]), float(slewed[1]))
        out.antennas = (float(slewed[0]), float(slewed[1]))
        # Eyes.
        self.robot.set_eyes(int(show.eyes))
        # Discrete events -> sounds / projector / eyes (never rate-limited).
        for ev in show.events:
            if ev.type == "sound":
                self.robot.play_sound(ev.value)
            elif ev.type == "projector":
                self.robot.set_projector(ev.value in ("on", "1", "true", "True"))
            elif ev.type == "eye":
                # Expressive eye cue (wide/blink/happy). Optional on the robot
                # interface: only RealRobot implements it, mocks simply skip.
                fn = getattr(self.robot, "set_eye_event", None)
                if fn is not None:
                    fn(ev.value)
            out.events_fired.append(ev)
