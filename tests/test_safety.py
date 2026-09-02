"""Unit tests for the minimum safety set (plan §6.5): watchdogs, thermal/duty,
e-stop, and the fault-action decision."""

import numpy as np
import pytest

from mini_bdx_runtime.anim.safety import (
    SafetyMonitor, SafetyConfig, FaultAction,
    DeadlineWatchdog, StaleCommandWatchdog, ThermalManager,
)
from mini_bdx_runtime.anim.hardware import SensorSnapshot, OperatorInput


def snap(t, temps=None, currents=None, **op):
    return SensorSnapshot(
        t_monotonic=t, joint_positions=np.zeros(14), joint_velocities=np.zeros(14),
        tilt_rad=0.0, feet_contacts=np.array([1.0, 1.0]),
        temperatures_c=temps, currents_a=currents, operator=OperatorInput(**op),
    )


# --- deadline watchdog ------------------------------------------------------
def test_deadline_tolerates_single_late_tick():
    wd = DeadlineWatchdog(SafetyConfig())
    for _ in range(50):
        assert wd.record(0.02) is None
    # One late tick within tolerance budget is fine.
    assert wd.record(0.029) is None


def test_deadline_faults_on_consecutive_overruns():
    cfg = SafetyConfig(deadline_max_consecutive=5)
    wd = DeadlineWatchdog(cfg)
    reason = None
    for _ in range(5):
        reason = wd.record(0.05)  # way over 0.02*(1.5)=0.03
    assert reason is not None
    assert "consecutive" in reason


def test_deadline_faults_on_high_miss_rate():
    cfg = SafetyConfig(deadline_window=20, deadline_max_miss_rate=0.5,
                       deadline_max_consecutive=1000)
    wd = DeadlineWatchdog(cfg)
    reason = None
    # Alternate hit/miss -> 50% exactly is not > 0.5; make 60% misses.
    for i in range(20):
        dur = 0.05 if (i % 5 != 0) else 0.02   # 16/20 misses
        reason = wd.record(dur)
    assert reason is not None
    assert "miss rate" in reason


# --- stale-command watchdog -------------------------------------------------
def test_stale_then_lost():
    cfg = SafetyConfig(stale_timeout_s=0.5, lost_timeout_s=1.0)
    wd = StaleCommandWatchdog(cfg)
    wd.heartbeat(0.0)
    stale, fault = wd.update(0.2, balancing=True)
    assert not stale and fault is None
    stale, fault = wd.update(0.7, balancing=True)
    assert stale and fault is None      # stale, fall back to idle
    stale, fault = wd.update(1.5, balancing=True)
    assert fault is not None            # persistent loss -> fault


def test_lost_does_not_fault_in_dock_by_default():
    cfg = SafetyConfig(stale_timeout_s=0.5, lost_timeout_s=1.0,
                       lost_faults_in_balancing_only=True)
    wd = StaleCommandWatchdog(cfg)
    wd.heartbeat(0.0)
    stale, fault = wd.update(2.0, balancing=False)   # dock: idle, no fault
    assert stale and fault is None


# --- thermal / duty ---------------------------------------------------------
def test_thermal_duty_limit_requests_cooldown():
    cfg = SafetyConfig(max_continuous_load_s=1.0, cooldown_s=0.5)
    tm = ThermalManager(cfg)
    cooldown = False
    for _ in range(60):  # 60 * 0.02 = 1.2s under load
        _, cooldown = tm.update(snap(0.0), under_load=True, dt=0.02)
    assert cooldown is True


def test_thermal_cooldown_releases_after_decay():
    cfg = SafetyConfig(max_continuous_load_s=1.0, cooldown_s=0.5,
                       cooldown_decay_rate=1.0)
    tm = ThermalManager(cfg)
    for _ in range(60):
        tm.update(snap(0.0), under_load=True, dt=0.02)
    assert tm.cooling
    # Relieve load; load_time must decay below the release threshold.
    for _ in range(60):
        _, cooldown = tm.update(snap(0.0), under_load=False, dt=0.02)
    assert tm.cooling is False


def test_thermal_hard_temp_limit_faults_when_available():
    cfg = SafetyConfig(temp_limit_c=65.0)
    tm = ThermalManager(cfg)
    temps = np.full(14, 70.0)
    reason, cooldown = tm.update(snap(0.0, temps=temps), under_load=True, dt=0.02)
    assert reason is not None and "temperature" in reason
    assert cooldown is True


def test_thermal_temp_none_is_time_based_only():
    tm = ThermalManager(SafetyConfig())
    reason, _ = tm.update(snap(0.0, temps=None), under_load=True, dt=0.02)
    assert reason is None
    assert tm.temp_available is False


# --- safety monitor + fault action ------------------------------------------
def test_estop_sets_fault_with_default_action():
    mon = SafetyMonitor(SafetyConfig(default_fault_action=FaultAction.CONTROLLED_HOLD))
    st = mon.update(snap(0.0, estop=True), balancing=True, under_load=True, dt=0.02)
    assert st.fault and st.estop
    assert st.fault_action == FaultAction.CONTROLLED_HOLD


def test_thermal_fault_forces_torque_off():
    mon = SafetyMonitor(SafetyConfig(
        default_fault_action=FaultAction.CONTROLLED_HOLD,
        thermal_fault_forces_torque_off=True, temp_limit_c=65.0))
    temps = np.full(14, 80.0)
    st = mon.update(snap(0.0, temps=temps), balancing=True, under_load=True, dt=0.02)
    assert st.fault
    assert st.fault_action == FaultAction.TORQUE_OFF  # thermal overrides hold


def test_no_fault_when_healthy():
    mon = SafetyMonitor(SafetyConfig())
    mon.heartbeat(0.0)
    st = mon.update(snap(0.01), balancing=True, under_load=True, dt=0.02,
                    last_tick_duration_s=0.02)
    assert not st.fault


def test_deadline_via_monitor_faults():
    cfg = SafetyConfig(deadline_max_consecutive=3)
    mon = SafetyMonitor(cfg)
    mon.heartbeat(0.0)
    st = None
    for i in range(3):
        st = mon.update(snap(i * 0.02), balancing=True, under_load=True, dt=0.02,
                        last_tick_duration_s=0.05)
    assert st.fault and "deadline" in st.fault_reason
