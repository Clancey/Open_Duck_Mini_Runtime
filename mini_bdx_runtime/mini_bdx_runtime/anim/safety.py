"""Minimum safety set for the animation runtime (plan §6.5, Phase 3/4).

Ships in the first hardware-touching phase, not later. Provides:

* **Controlled clip abort** — handled by the ``open_duck_anim`` Engine itself
  (``Triggers(cancel=True)`` blends head/show back to the owning controller over
  ``T_alpha`` / ``T_beta``, never an instantaneous cut). The controller exposes
  it; this module records the intent for logging.
* **Deadman / e-stop -> latched FAULT** — the FSM latches; this module chooses
  the **fault action** (torque-off vs controlled-hold) and drives antennas to
  neutral, sounds/projector off.
* **Watchdogs** — :class:`DeadlineWatchdog` (control-loop deadline miss) and
  :class:`StaleCommandWatchdog` (lost controller / stale command).
* **Thermal / load** — :class:`ThermalManager` reads Feetech temperature/current
  *where the HWI exposes it* (``rustypot==0.1.0`` does NOT, see
  :mod:`.hardware`), enforces a per-servo limit when present, and always
  enforces a **max continuous demo duration + cooldown** with a load-relieving
  dock posture.

FAULT ACTION CHOICE (plan §6.5-b / Q6), with reasoning:

    Default = **CONTROLLED_HOLD**. For a small standing biped, torque-off makes
    the robot go limp and *guarantees* a fall/flop, which risks mechanical
    damage (the runtime README warns the head can break) and is more dangerous
    than briefly holding the last commanded pose. Holding keeps the servos
    energised and the posture intact while an operator intervenes.

    EXCEPTION = a **thermal/overload** fault forces **TORQUE_OFF**. Holding pose
    under torque is the one case where "hold" is actively harmful: it sustains
    the stall current that caused the overheat. So thermal faults cut torque
    (and the load-relieving dock posture is the recovery), overriding the
    default. This is configurable via :class:`SafetyConfig`.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

from .hardware import SensorSnapshot


class FaultAction(Enum):
    TORQUE_OFF = "torque_off"
    CONTROLLED_HOLD = "controlled_hold"


@dataclass
class SafetyConfig:
    # --- fault action ---
    default_fault_action: FaultAction = FaultAction.CONTROLLED_HOLD
    thermal_fault_forces_torque_off: bool = True

    # --- deadline-miss watchdog (plan §6.5-c-i) ---
    ctrl_dt_s: float = 0.02
    deadline_tolerance_frac: float = 0.5   # a tick over budget*(1+tol) is a miss
    deadline_max_consecutive: int = 10     # consecutive severe misses -> fault
    deadline_window: int = 200             # rolling window for the miss rate
    deadline_max_miss_rate: float = 0.5    # >50% misses in the window -> fault

    # --- stale-command / lost-controller watchdog (plan §6.5-c-ii) ---
    stale_timeout_s: float = 0.5           # no heartbeat -> fall back to idle
    lost_timeout_s: float = 3.0            # persistent loss -> fault
    lost_faults_in_balancing_only: bool = True  # DOCK loss just idles; STAND/WALK faults

    # --- thermal / load (plan §6.5-d) ---
    temp_limit_c: float = 65.0             # per-servo hard limit (STS3215 spec ~70C)
    current_limit_a: float = 2.0           # per-servo hard current limit (advisory)
    max_continuous_load_s: float = 120.0   # max continuous demo duration under load
    cooldown_s: float = 60.0               # cooldown before re-enabling full load
    cooldown_decay_rate: float = 1.0       # load-time decay per second while relieved


# ---------------------------------------------------------------------------
# Watchdogs
# ---------------------------------------------------------------------------
@dataclass
class DeadlineWatchdog:
    """Control-loop deadline-miss watchdog (plan §6.5-c-i).

    Call :meth:`record` with each measured tick duration. A tick over
    ``ctrl_dt*(1+tolerance)`` is a *miss*. Faults on either too many consecutive
    severe misses or a too-high miss rate over a rolling window — so a single
    late tick (a GC pause) is tolerated but sustained overruns are not.
    """

    cfg: SafetyConfig
    consecutive: int = 0
    total_misses: int = 0
    total_ticks: int = 0
    worst_overrun_s: float = 0.0
    _window: List[int] = field(default_factory=list)

    def record(self, tick_duration_s: float) -> Optional[str]:
        budget = self.cfg.ctrl_dt_s * (1.0 + self.cfg.deadline_tolerance_frac)
        miss = tick_duration_s > budget
        self.total_ticks += 1
        overrun = tick_duration_s - self.cfg.ctrl_dt_s
        if overrun > self.worst_overrun_s:
            self.worst_overrun_s = overrun
        if miss:
            self.consecutive += 1
            self.total_misses += 1
        else:
            self.consecutive = 0
        self._window.append(1 if miss else 0)
        if len(self._window) > self.cfg.deadline_window:
            self._window.pop(0)

        if self.consecutive >= self.cfg.deadline_max_consecutive:
            return ("deadline watchdog: %d consecutive control-loop overruns"
                    % self.consecutive)
        if len(self._window) >= self.cfg.deadline_window:
            rate = sum(self._window) / float(len(self._window))
            if rate > self.cfg.deadline_max_miss_rate:
                return ("deadline watchdog: miss rate %.0f%% over %d ticks"
                        % (rate * 100.0, len(self._window)))
        return None


@dataclass
class StaleCommandWatchdog:
    """Stale-command / lost-controller watchdog (plan §6.5-c-ii).

    :meth:`heartbeat` on every fresh trigger/command. :meth:`update` each tick
    returns ``(stale, fault_reason)`` where ``stale`` means "fall back to
    background idle" and ``fault_reason`` (non-None) means persistent loss ->
    fault. In non-balancing modes (DOCK) a lost controller only idles unless
    configured otherwise; in STAND/WALK persistent loss faults.
    """

    cfg: SafetyConfig
    _last_heartbeat_t: Optional[float] = None

    def heartbeat(self, t: float) -> None:
        self._last_heartbeat_t = t

    def update(self, t: float, balancing: bool):
        if self._last_heartbeat_t is None:
            self._last_heartbeat_t = t
            return False, None
        age = t - self._last_heartbeat_t
        stale = age > self.cfg.stale_timeout_s
        fault = None
        if age > self.cfg.lost_timeout_s:
            if balancing or not self.cfg.lost_faults_in_balancing_only:
                fault = ("stale-command watchdog: no controller heartbeat for "
                         "%.1f s" % age)
        return stale, fault


@dataclass
class ThermalManager:
    """Thermal / load management (plan §6.5-d).

    ``rustypot==0.1.0`` does not expose STS3215 temperature/current, so
    :attr:`SensorSnapshot.temperatures_c` / ``currents_a`` are ``None`` on
    hardware today. When they ARE present (future HWI), a per-servo limit faults
    immediately. Independent of that, a **time-based** duty limit always applies:
    continuous load time accumulates in STAND/WALK and decays while relieved
    (DOCK / disarmed). Past ``max_continuous_load_s`` the manager requests a
    cooldown (the caller drops to the load-relieving dock posture); a hard
    temperature reading past the limit faults.
    """

    cfg: SafetyConfig
    load_time_s: float = 0.0
    cooling: bool = False
    peak_temp_c: float = 0.0
    peak_current_a: float = 0.0
    temp_available: bool = False

    def update(self, snap: SensorSnapshot, under_load: bool, dt: float):
        """Return ``(fault_reason_or_None, request_cooldown_bool)``."""
        # Hard per-servo temperature limit (only if the HWI exposes it).
        if snap.temperatures_c is not None:
            self.temp_available = True
            tmax = float(np.max(snap.temperatures_c))
            if tmax > self.peak_temp_c:
                self.peak_temp_c = tmax
            if tmax > self.cfg.temp_limit_c:
                return ("thermal: servo temperature %.1f C > limit %.1f C"
                        % (tmax, self.cfg.temp_limit_c)), True
        if snap.currents_a is not None:
            imax = float(np.max(np.abs(snap.currents_a)))
            if imax > self.peak_current_a:
                self.peak_current_a = imax
            if imax > self.cfg.current_limit_a:
                return ("thermal: servo current %.2f A > limit %.2f A"
                        % (imax, self.cfg.current_limit_a)), True

        # Time-based duty accumulator (always active).
        if under_load:
            self.load_time_s += dt
        else:
            self.load_time_s = max(
                0.0, self.load_time_s - self.cfg.cooldown_decay_rate * dt
            )

        # Cooldown hysteresis: request cooldown once over budget, keep requesting
        # until load time has decayed below the cooldown threshold.
        if self.load_time_s >= self.cfg.max_continuous_load_s:
            self.cooling = True
        elif self.load_time_s <= max(0.0, self.cfg.max_continuous_load_s - self.cfg.cooldown_s):
            self.cooling = False
        return None, self.cooling


# ---------------------------------------------------------------------------
# Safety monitor
# ---------------------------------------------------------------------------
@dataclass
class SafetyStatus:
    fault: bool = False
    fault_reason: str = ""
    fault_action: FaultAction = FaultAction.CONTROLLED_HOLD
    stale: bool = False
    request_cooldown: bool = False
    estop: bool = False


class SafetyMonitor:
    """Aggregates the watchdogs, thermal manager and e-stop (plan §6.5).

    Owns the fault-action decision. Wire it into the control loop alongside the
    :class:`~mini_bdx_runtime.anim.fsm.ModeFSM`: the monitor detects faults and
    the FSM latches them.
    """

    def __init__(self, config: Optional[SafetyConfig] = None):
        self.cfg = config or SafetyConfig()
        self.deadline = DeadlineWatchdog(self.cfg)
        self.stale = StaleCommandWatchdog(self.cfg)
        self.thermal = ThermalManager(self.cfg)
        self._thermal_fault_latched = False

    def heartbeat(self, t: float) -> None:
        self.stale.heartbeat(t)

    def record_tick_duration(self, tick_duration_s: float) -> Optional[str]:
        return self.deadline.record(tick_duration_s)

    def update(
        self,
        snap: SensorSnapshot,
        balancing: bool,
        under_load: bool,
        dt: float,
        last_tick_duration_s: Optional[float] = None,
    ) -> SafetyStatus:
        status = SafetyStatus()
        status.estop = bool(snap.operator.estop)

        reasons: List[str] = []
        thermal_fault = False

        if snap.operator.estop:
            reasons.append("e-stop asserted")

        if last_tick_duration_s is not None:
            dl = self.deadline.record(last_tick_duration_s)
            if dl:
                reasons.append(dl)

        stale, lost = self.stale.update(snap.t_monotonic, balancing)
        status.stale = stale
        if lost:
            reasons.append(lost)

        th_reason, cooldown = self.thermal.update(snap, under_load, dt)
        status.request_cooldown = cooldown
        if th_reason:
            reasons.append(th_reason)
            thermal_fault = True

        if reasons:
            status.fault = True
            status.fault_reason = "; ".join(reasons)
            # Fault-action decision: thermal overrides to torque-off.
            if thermal_fault and self.cfg.thermal_fault_forces_torque_off:
                status.fault_action = FaultAction.TORQUE_OFF
            else:
                status.fault_action = self.cfg.default_fault_action
        else:
            status.fault_action = self.cfg.default_fault_action
        return status
