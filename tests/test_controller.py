"""Unit tests for the AnimationController integration (plan §6.1-6.5):
single engine eval per tick, capability matrix, additive head path, controlled
abort/blend-back, the final bus velocity clip and the fault actions."""

import os

import numpy as np
import pytest

from mini_bdx_runtime.anim.controller import AnimationController, ControllerConfig
from mini_bdx_runtime.anim.hardware import MockRobot, SensorSnapshot, OperatorInput
from mini_bdx_runtime.anim.fsm import FSMState
from mini_bdx_runtime.anim.safety import SafetyConfig, FaultAction

_CLIP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "clips", "idle_alive.duckanim")

INIT = np.array([0.002, 0.053, -0.63, 1.368, -0.784, 0.0, 0.0, 0.0, 0.0,
                 -0.003, -0.065, 0.635, 1.379, -0.796])


def _bg():
    from mini_bdx_runtime.anim import _ensure_open_duck_anim
    _ensure_open_duck_anim()
    from open_duck_anim import load_clip
    return load_clip(_CLIP)


def make(**cfg):
    robot = MockRobot()
    c = AnimationController(robot, background_clip=_bg(),
                            config=ControllerConfig(**cfg), init_pos_14=INIT)
    return robot, c


def setsnap(robot, t, pos=None, vel=0.0, tilt=0.0, feet=(1.0, 1.0)):
    if pos is None:
        pos = INIT
    robot.next_snapshot = SensorSnapshot(
        t_monotonic=t, joint_positions=np.asarray(pos, float),
        joint_velocities=np.full(14, vel) if np.isscalar(vel) else np.asarray(vel, float),
        tilt_rad=tilt, feet_contacts=np.asarray(feet, float),
    )


def force_state(c, state):
    c.fsm.state = state
    c.fsm._last_t = None


# --- engine evaluated exactly once per tick ---------------------------------
def test_engine_evaluated_once_per_tick(monkeypatch):
    robot, c = make()
    force_state(c, FSMState.DOCK_DEMO)
    calls = {"n": 0}
    real = c.engine.evaluate

    def spy(t, mode, trg=None):
        calls["n"] += 1
        return real(t, mode, trg)
    monkeypatch.setattr(c.engine, "evaluate", spy)
    setsnap(robot, 0.1)
    c.step(OperatorInput())
    assert calls["n"] == 1


def test_engine_not_evaluated_in_boot():
    robot, c = make()
    # BOOT by default.
    setsnap(robot, 0.1)
    plan = c.prepare(OperatorInput())
    out = c.finalize(None)
    assert not out.engine_evaluated
    assert out.torque_off and out.bus_targets is None
    assert robot.torque_is_off


# --- DOCK_DEMO: legs held at init_pos, head animated ------------------------
def test_dock_holds_legs_at_init_and_animates_head():
    robot, c = make()
    force_state(c, FSMState.DOCK_DEMO)
    leg_idx = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
    last = None
    for i in range(30):
        setsnap(robot, 0.1 + i * 0.02, pos=(last if last is not None else INIT))
        out = c.step(OperatorInput())
        last = out.bus_targets
    # Legs stay pinned at init_pos (dock hold), never driven by animation.
    np.testing.assert_allclose(out.bus_targets[leg_idx], INIT[leg_idx], atol=1e-6)
    # Head moved away from zero (the idle clip animates it).
    assert np.any(np.abs(out.bus_targets[5:9]) > 1e-4)
    # Antennas and eyes were driven from show_functions.
    assert len(robot.antenna_history) > 0
    assert len(robot.eye_history) > 0


# --- capability matrix: legs discarded in STAND/WALK ------------------------
def test_stand_discards_animated_legs_uses_policy():
    robot, c = make()
    force_state(c, FSMState.STAND)
    setsnap(robot, 0.1)
    c.prepare(OperatorInput())
    # Policy commands distinct leg targets; head from policy too.
    policy = INIT.copy()
    policy[3] = 1.10   # left_knee commanded well below init
    out = c.finalize(policy)
    leg_idx = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
    # Legs come from the policy (rate-limited toward it), not from animation.
    # The commanded left_knee moved toward the policy value, below init.
    assert out.bus_targets[3] < INIT[3]


def test_additive_head_path_adds_offset_to_policy():
    robot, c = make(enable_rate_limit=False)
    force_state(c, FSMState.STAND)
    setsnap(robot, 0.1)
    plan = c.prepare(OperatorInput())
    offset = plan.head_offsets.copy()
    assert np.any(np.abs(offset) > 1e-4)  # clip produced a real head offset
    policy = INIT.copy()
    out = c.finalize(policy)
    # bus head == policy head + engine offset (clamped by joint limits only).
    expected = policy[5:9] + offset
    np.testing.assert_allclose(out.bus_targets[5:9], expected, atol=1e-6)


# --- controlled abort / blend-back ------------------------------------------
def test_cooldown_injects_cancel_for_controlled_abort(monkeypatch):
    robot, c = make()
    force_state(c, FSMState.DOCK_DEMO)
    seen = {}
    real = c.engine.evaluate

    def spy(t, mode, trg=None):
        seen["cancel"] = getattr(trg, "cancel", None)
        return real(t, mode, trg)
    monkeypatch.setattr(c.engine, "evaluate", spy)
    # Force the thermal manager into cooldown so the controller requests abort.
    c.safety.thermal.load_time_s = 999.0   # above the duty limit -> stays cooling
    setsnap(robot, 0.1)
    c.step(OperatorInput())
    assert seen["cancel"] is True   # active clips cancelled -> blend back


# --- final bus velocity clip (5.24 rad/s) re-enabled ------------------------
def test_final_velocity_clip_bounds_step():
    robot, c = make()
    dt = c.cfg.ctrl_dt
    c._prev_bus_targets = INIT.copy()
    big = INIT.copy()
    big[7] += 5.0   # huge head_yaw jump
    clipped = c._apply_final_safety(big)
    max_step = c.cfg.max_motor_velocity * dt
    assert abs(clipped[7] - INIT[7]) <= max_step + 1e-9


def test_joint_position_clamp_applied():
    robot, c = make(enable_rate_limit=False)
    over = INIT.copy()
    over[7] = 10.0   # head_yaw beyond +2.7925 limit
    clamped = c._apply_final_safety(over)
    assert clamped[7] <= 2.7925 + 1e-9


# --- fault actions ----------------------------------------------------------
def test_fault_controlled_hold_keeps_last_pose():
    robot, c = make(envelope_derating=0.5)
    c._prev_bus_targets = INIT.copy()
    c.fsm.to_fault("boom")
    setsnap(robot, 0.2)
    out = c.finalize(None) if c.prepare(OperatorInput()) else None
    assert out is not None
    assert not robot.torque_is_off               # held, not limp
    np.testing.assert_allclose(out.bus_targets, INIT, atol=1e-9)
    assert robot.show_shutdown_count >= 1        # show neutralised


def test_fault_torque_off_action():
    robot, c = make()
    c.safety.cfg.default_fault_action = FaultAction.TORQUE_OFF
    c._prev_bus_targets = INIT.copy()
    c.fsm.to_fault("boom")
    setsnap(robot, 0.2)
    c.prepare(OperatorInput())
    out = c.finalize(None)
    assert robot.torque_is_off
    assert out.torque_off


def test_thermal_fault_forces_torque_off_end_to_end():
    robot, c = make()
    force_state(c, FSMState.STAND)
    c._prev_bus_targets = INIT.copy()
    temps = np.full(14, 90.0)
    robot.next_snapshot = SensorSnapshot(
        t_monotonic=0.3, joint_positions=INIT, joint_velocities=np.zeros(14),
        tilt_rad=0.0, feet_contacts=np.array([1.0, 1.0]), temperatures_c=temps,
    )
    c.prepare(OperatorInput())
    out = c.finalize(INIT.copy())
    assert c.fsm.state == FSMState.FAULT
    assert robot.torque_is_off   # thermal -> torque off regardless of default


# --- transient read failure skips the tick ----------------------------------
def test_read_failure_returns_none():
    robot, c = make()
    robot.fail_read = True
    assert c.prepare(OperatorInput()) is None


# --- shared head-engine factory (single source of truth for envelope+limits) --
def test_make_head_engine_is_envelope_clamped_and_shared():
    """The RL-walk overlay path and the AnimationController must build the head
    Engine identically: derated measured envelope + physical head joint limits.
    A large commanded head deflection must be clamped to the derated budget."""
    from mini_bdx_runtime.anim.controller import make_head_engine
    from mini_bdx_runtime.anim import _ensure_open_duck_anim
    _ensure_open_duck_anim()
    from open_duck_anim import Triggers

    eng = make_head_engine(envelope_derating=0.5)
    # Drive a big head-yaw offset via a joystick trigger; envelope must clamp it.
    trg = Triggers(joystick_offset=np.array([0.0, 0.0, 5.0, 0.0]))
    out = eng.evaluate(0.02, "walk", trg)
    off = np.asarray(out.head_command_offsets, dtype=float)
    assert off.shape == (4,)
    # Derated head_yaw per-channel limit is ~0.75 rad; must not exceed it.
    assert abs(off[2]) <= 0.75 + 1e-6
    assert not eng.head_fault  # a clamped command is not a fault


def test_make_head_engine_none_derating_uses_full_envelope():
    from mini_bdx_runtime.anim.controller import make_head_engine
    from mini_bdx_runtime.anim import _ensure_open_duck_anim
    _ensure_open_duck_anim()
    from open_duck_anim import Triggers

    derated = make_head_engine(envelope_derating=0.5)
    full = make_head_engine(envelope_derating=None)
    trg = Triggers(joystick_offset=np.array([0.0, 0.0, 5.0, 0.0]))
    d = abs(float(np.asarray(derated.evaluate(0.02, "walk", trg).head_command_offsets)[2]))
    f = abs(float(np.asarray(full.evaluate(0.02, "walk", trg).head_command_offsets)[2]))
    assert f > d  # full envelope allows a larger deflection than derated


# --- eye events reach the device through _drive_show (regression for bug #2) --
def test_eye_event_in_show_reaches_device_via_drive_show():
    """_drive_show used to handle only 'sound' and 'projector' events and
    silently drop 'eye' events, so clip eye cues (wide/blink/happy) never reached
    the hardware. An 'eye' event authored in a clip's show_functions.events must
    now be routed to the robot's set_eye_event."""
    import types
    from open_duck_anim import TickShow, DiscreteEvent

    robot, c = make()
    show = TickShow(
        antenna_l=0.0, antenna_r=0.0, eyes=1,
        events=[DiscreteEvent(frame=0, type="eye", value="wide")],
    )
    engine_out = types.SimpleNamespace(show=show)
    out = types.SimpleNamespace(antennas=None, events_fired=[])
    c._drive_show(engine_out, out)
    # The eye cue reached the device and the event was recorded as fired.
    assert robot.eye_event_history == ["wide"]
    assert [ev.type for ev in out.events_fired] == ["eye"]


def test_mixed_show_events_route_to_correct_channels():
    """sound/projector/eye events each go to their own channel, in order."""
    import types
    from open_duck_anim import TickShow, DiscreteEvent

    robot, c = make()
    show = TickShow(
        antenna_l=0.0, antenna_r=0.0, eyes=1,
        events=[
            DiscreteEvent(frame=0, type="sound", value="beep"),
            DiscreteEvent(frame=0, type="eye", value="blink"),
            DiscreteEvent(frame=0, type="projector", value="on"),
        ],
    )
    engine_out = types.SimpleNamespace(show=show)
    out = types.SimpleNamespace(antennas=None, events_fired=[])
    c._drive_show(engine_out, out)
    assert robot.sound_history == ["beep"]
    assert robot.eye_event_history == ["blink"]
    assert robot.projector_state is True
