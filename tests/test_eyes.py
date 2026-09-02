"""Composition tests for the eye background/idle blink logic.

The real :class:`mini_bdx_runtime.eyes.Eyes` drives two GPIO pins via CircuitPython
``board``/``digitalio``. Those aren't present off-robot, so we inject tiny fakes
that record every pin transition, then exercise the *real* Eyes background thread
to prove the three-layer composition the hardware validated:

* eyes are lit at baseline,
* an expressive clip cue takes precedence over the idle loop (``hold_open``
  suppresses idle blinking; ``blink`` injects one immediately), and
* the idle blink resumes automatically once the cue window passes.
"""

import sys
import time
import types

import pytest


class _FakePin:
    def __init__(self, name):
        self.name = name
        self.value = None
        self.direction = None
        self.deinited = False
        self.transitions = []  # (monotonic, bool) each time value is written

    # digitalio.DigitalInOut(pin) stores the pin; value is a property in the
    # real lib, but a plain attribute works for our recording needs via a
    # wrapper below.


class _RecordingDIO:
    """Stand-in for digitalio.DigitalInOut that records value transitions."""

    def __init__(self, pin):
        self._pin = pin
        self.direction = None

    @property
    def value(self):
        return self._pin.value

    @value.setter
    def value(self, v):
        self._pin.value = bool(v)
        self._pin.transitions.append((time.monotonic(), bool(v)))

    def deinit(self):
        self._pin.deinited = True


@pytest.fixture
def eyes_module(monkeypatch):
    """Inject fake board/digitalio and import a fresh Eyes each test."""
    left = _FakePin("D24")
    right = _FakePin("D23")

    board = types.ModuleType("board")
    board.D24 = left
    board.D23 = right

    digitalio = types.ModuleType("digitalio")
    digitalio.DigitalInOut = _RecordingDIO
    direction = types.SimpleNamespace(OUTPUT="output", INPUT="input")
    digitalio.Direction = direction

    monkeypatch.setitem(sys.modules, "board", board)
    monkeypatch.setitem(sys.modules, "digitalio", digitalio)
    # Force a fresh import so module-level board.D24/D23 bind to our fakes.
    monkeypatch.delitem(sys.modules, "mini_bdx_runtime.eyes", raising=False)
    import mini_bdx_runtime.eyes as eyes_mod
    yield eyes_mod, left, right


def _darks_between(pin, t0, t1):
    """Count dark (value==False) transitions in the half-open window (t0, t1]."""
    return sum(1 for (ts, v) in pin.transitions if t0 < ts <= t1 and v is False)


def test_eyes_lit_at_baseline(eyes_module):
    eyes_mod, left, right = eyes_module
    # Long idle interval so no idle blink fires during the check.
    e = eyes_mod.Eyes(min_interval=100.0, max_interval=100.0)
    try:
        time.sleep(0.1)
        assert left.value is True and right.value is True   # resting = lit/open
    finally:
        e.stop()


def test_clip_cue_takes_precedence_over_idle_then_idle_resumes(eyes_module):
    """hold_open suppresses the idle blink for its window; once it lapses the
    background idle blink resumes on its own."""
    eyes_mod, left, right = eyes_module
    # Fast idle blink so, absent a cue, dark flicks would happen frequently.
    e = eyes_mod.Eyes(blink_duration=0.02, min_interval=0.08, max_interval=0.08,
                      double_blink_prob=0.0)
    try:
        time.sleep(0.05)                 # let the thread settle, eyes lit
        t_hold = time.monotonic()
        e.hold_open(0.4)                 # clip cue: stay wide, suppress idle
        time.sleep(0.3)                  # well inside the hold window
        # No dark flicks during the hold: the cue took precedence over idle.
        assert _darks_between(left, t_hold, time.monotonic()) == 0
        assert left.value is True        # eyes held open

        t_resume = time.monotonic()
        time.sleep(0.5)                  # past the 0.4 s hold -> idle resumes
        # Idle blink resumed automatically after the cue window.
        assert _darks_between(left, t_resume + 0.05, time.monotonic()) >= 1
    finally:
        e.stop()


def test_blink_cue_injected_immediately(eyes_module):
    """An explicit blink() cue fires now, not on the far-off idle schedule."""
    eyes_mod, left, right = eyes_module
    # Idle is 5 s away, so any dark flick within 0.2 s must be the injected cue.
    e = eyes_mod.Eyes(blink_duration=0.02, min_interval=5.0, max_interval=5.0,
                      double_blink_prob=0.0)
    try:
        time.sleep(0.05)
        t0 = time.monotonic()
        e.blink()
        time.sleep(0.15)
        assert _darks_between(left, t0, time.monotonic()) >= 1
        assert left.value is True        # returns to lit after the flick
    finally:
        e.stop()


def test_authored_falling_edge_triggers_blink(eyes_module):
    """note_authored honours a 1->0 authored edge as a blink (does not force
    the eyes dark), matching set_eyes() routing in real_robot."""
    eyes_mod, left, right = eyes_module
    e = eyes_mod.Eyes(blink_duration=0.02, min_interval=5.0, max_interval=5.0,
                      double_blink_prob=0.0)
    try:
        time.sleep(0.05)
        t0 = time.monotonic()
        e.note_authored(1)               # steady open, no edge
        time.sleep(0.1)
        assert _darks_between(left, t0, time.monotonic()) == 0
        t1 = time.monotonic()
        e.note_authored(0)               # 1->0 edge = one authored blink
        time.sleep(0.15)
        assert _darks_between(left, t1, time.monotonic()) >= 1
        assert left.value is True        # not left dark
    finally:
        e.stop()
