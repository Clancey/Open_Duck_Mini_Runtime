"""Unit tests for the mode FSM (plan §4.3): every transition, guard rejection,
the latched fault and the arming ramp. Pure, no robot."""

import numpy as np
import pytest

from mini_bdx_runtime.anim.fsm import ModeFSM, FSMState, FSMConfig
from mini_bdx_runtime.anim.hardware import SensorSnapshot, OperatorInput

INIT = np.array([0.002, 0.053, -0.63, 1.368, -0.784, 0.0, 0.0, 0.0, 0.0,
                 -0.003, -0.065, 0.635, 1.379, -0.796])


def snap(t, pos, vel=0.0, tilt=0.0, feet=(1.0, 1.0), **op):
    pos = np.asarray(pos, dtype=float)
    if np.isscalar(vel):
        vel = np.full(14, vel, dtype=float)
    return SensorSnapshot(
        t_monotonic=t, joint_positions=pos,
        joint_velocities=np.asarray(vel, dtype=float), tilt_rad=tilt,
        feet_contacts=np.asarray(feet, dtype=float),
        operator=OperatorInput(**op),
    )


def make():
    return ModeFSM(INIT, FSMConfig())


def arm_to_ready(fsm, t0=0.0, dt=0.02, **op):
    """Drive BOOT -> ARMING and run the ramp to completion with perfect
    tracking. Returns the time after the ramp is done. Extra operator kwargs are
    applied on every tick (e.g. dock_confirmed)."""
    t = t0
    fsm.update(snap(t, INIT, arm_requested=True), INIT)
    assert fsm.state == FSMState.ARMING
    # Run past t_arm_s with measured following the ramp reference.
    n = int((fsm.cfg.t_arm_s + fsm.cfg.dwell_s + 0.2) / dt) + 5
    for _ in range(n):
        t += dt
        ref = fsm.arming_reference(t)
        fsm.update(snap(t, ref, vel=0.0, tilt=0.0, **op), ref)
    return t


# --- BOOT -------------------------------------------------------------------
def test_boot_stays_disarmed_without_arm():
    fsm = make()
    for i in range(5):
        st = fsm.update(snap(i * 0.02, INIT), INIT)
    assert fsm.state == FSMState.BOOT
    assert fsm.engine_mode() is None


def test_boot_to_arming_on_arm_request():
    fsm = make()
    fsm.update(snap(0.0, INIT, arm_requested=True), INIT)
    assert fsm.state == FSMState.ARMING


def test_boot_self_check_failure_faults():
    fsm = ModeFSM(INIT, FSMConfig(), self_check=lambda s: (False, "imu dead"))
    fsm.update(snap(0.0, INIT, arm_requested=True), INIT)
    assert fsm.state == FSMState.FAULT
    assert "self-check" in fsm.fault_reason


# --- ARMING ramp ------------------------------------------------------------
def test_arming_reference_ramps_measured_to_init():
    fsm = make()
    measured0 = INIT + 0.2  # start away from init
    fsm.update(snap(0.0, measured0, arm_requested=True), measured0)
    # At t0 the reference equals the measured pose (never snaps).
    np.testing.assert_allclose(fsm.arming_reference(0.0), measured0, atol=1e-9)
    # At t0 + t_arm it equals init_pos.
    np.testing.assert_allclose(fsm.arming_reference(fsm.cfg.t_arm_s), INIT, atol=1e-9)
    # kp fraction ramps from arm_kp_low_frac to 1.0.
    assert fsm.arming_kp_frac(0.0) == pytest.approx(fsm.cfg.arm_kp_low_frac)
    assert fsm.arming_kp_frac(fsm.cfg.t_arm_s) == pytest.approx(1.0)


def test_arming_tracking_mismatch_faults():
    fsm = make()
    fsm.update(snap(0.0, INIT, arm_requested=True), INIT)
    # Feed a measured pose far from the ramp reference -> mismatch fault.
    fsm.update(snap(0.02, INIT + 1.0), INIT)
    assert fsm.state == FSMState.FAULT
    assert "tracking mismatch" in fsm.fault_reason


def test_arming_timeout_faults():
    fsm = make()
    fsm.update(snap(0.0, INIT, arm_requested=True), INIT)
    t = 0.0
    # Track the ramp (no mismatch) but keep tilt high so guards never hold;
    # eventually the arming timeout fires.
    while fsm.state == FSMState.ARMING and t < 6.0:
        t += 0.02
        ref = fsm.arming_reference(t)
        fsm.update(snap(t, ref, tilt=1.0), ref)
    assert fsm.state == FSMState.FAULT
    assert "timeout" in fsm.fault_reason


def test_arming_holds_until_destination_confirmed():
    fsm = make()
    t = arm_to_ready(fsm)  # no dock/offdock confirm
    assert fsm.state == FSMState.ARMING  # armed, waiting for a destination


# --- ARMING -> operational --------------------------------------------------
def test_arming_to_dock_on_confirm():
    fsm = make()
    arm_to_ready(fsm, dock_confirmed=True)
    assert fsm.state == FSMState.DOCK_DEMO
    assert fsm.engine_mode() == "dock"


def test_arming_to_stand_on_offdock():
    fsm = make()
    arm_to_ready(fsm, offdock_confirmed=True)
    assert fsm.state == FSMState.STAND
    assert fsm.engine_mode() == "stand"


# --- operational transitions ------------------------------------------------
def _drive(fsm, t, n, pos, **op):
    for _ in range(n):
        t += 0.02
        fsm.update(snap(t, pos, vel=0.0, tilt=0.0, **op), pos)
    return t


def test_stand_to_walk_and_back():
    fsm = make()
    t = arm_to_ready(fsm, offdock_confirmed=True)
    assert fsm.state == FSMState.STAND
    # Nonzero locomotion command -> WALK.
    t = _drive(fsm, t, 2, INIT, locomotion_command=(0.3, 0.0, 0.0))
    assert fsm.state == FSMState.WALK
    assert fsm.engine_mode() == "walk"
    # Command returns to zero -> STAND.
    t = _drive(fsm, t, 2, INIT)
    assert fsm.state == FSMState.STAND


def test_stand_to_dock_requires_guards():
    fsm = make()
    t = arm_to_ready(fsm, offdock_confirmed=True)
    assert fsm.state == FSMState.STAND
    # Dock confirm with guards satisfied (still, upright, both feet) over dwell.
    t = _drive(fsm, t, 25, INIT, dock_confirmed=True)
    assert fsm.state == FSMState.DOCK_DEMO


def test_dock_to_stand_on_offdock():
    fsm = make()
    t = arm_to_ready(fsm, dock_confirmed=True)
    assert fsm.state == FSMState.DOCK_DEMO
    t = _drive(fsm, t, 25, INIT, offdock_confirmed=True)
    assert fsm.state == FSMState.STAND


# --- guard rejections -------------------------------------------------------
def test_operational_tilt_faults_in_stand():
    fsm = make()
    t = arm_to_ready(fsm, offdock_confirmed=True)
    fsm.update(snap(t + 0.02, INIT, tilt=0.3), INIT)  # > operational_tilt 0.15
    assert fsm.state == FSMState.FAULT
    assert "tilt" in fsm.fault_reason


def test_dock_transition_rejected_when_moving():
    fsm = make()
    t = arm_to_ready(fsm, offdock_confirmed=True)
    # High joint velocity blocks the STAND->DOCK guard indefinitely.
    t = _drive(fsm, t, 30, INIT, vel_hold=True) if False else t
    for _ in range(30):
        t += 0.02
        fsm.update(snap(t, INIT, vel=5.0, dock_confirmed=True), INIT)
    assert fsm.state == FSMState.STAND  # never satisfied the velocity guard


# --- e-stop / latched fault -------------------------------------------------
@pytest.mark.parametrize("reach", ["boot", "arming", "stand", "dock"])
def test_estop_latches_fault_from_any_state(reach):
    fsm = make()
    t = 0.0
    if reach == "arming":
        fsm.update(snap(0.0, INIT, arm_requested=True), INIT)
    elif reach == "stand":
        t = arm_to_ready(fsm, offdock_confirmed=True)
    elif reach == "dock":
        t = arm_to_ready(fsm, dock_confirmed=True)
    fsm.update(snap(t + 0.02, INIT, estop=True), INIT)
    assert fsm.state == FSMState.FAULT
    assert "e-stop" in fsm.fault_reason


def test_external_fault_latches():
    fsm = make()
    t = arm_to_ready(fsm, dock_confirmed=True)
    fsm.update(snap(t + 0.02, INIT), INIT, external_fault=True,
               external_fault_reason="watchdog")
    assert fsm.state == FSMState.FAULT
    assert fsm.fault_reason == "watchdog"


def test_fault_is_latched_and_not_self_clearing():
    fsm = make()
    fsm.to_fault("boom")
    # Even with perfect conditions and no estop, it stays FAULT.
    for i in range(10):
        fsm.update(snap(i * 0.02, INIT), INIT)
    assert fsm.state == FSMState.FAULT
    assert fsm.fault_reason == "boom"


def test_explicit_reset_returns_to_boot():
    fsm = make()
    fsm.to_fault("boom")
    fsm.update(snap(0.5, INIT), INIT)
    assert fsm.state == FSMState.FAULT
    ok = fsm.request_reset(snap(0.52, INIT, fault_reset=True))
    assert ok
    assert fsm.state == FSMState.BOOT
    assert fsm.fault_reason == ""


def test_engine_mode_none_in_nonoperational_states():
    fsm = make()
    assert fsm.engine_mode() is None          # BOOT
    fsm.update(snap(0.0, INIT, arm_requested=True), INIT)
    assert fsm.engine_mode() is None          # ARMING
    fsm.to_fault("x")
    assert fsm.engine_mode() is None          # FAULT
