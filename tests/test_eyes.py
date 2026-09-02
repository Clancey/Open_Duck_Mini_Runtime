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


def _max_dark_dwell(pin):
    """Longest interval the pin stayed dark (a False followed by a True)."""
    best = 0.0
    trans = pin.transitions
    for i in range(len(trans) - 1):
        ts, v = trans[i]
        if v is False:
            nts, nv = trans[i + 1]
            if nv is True:
                best = max(best, nts - ts)
    return best


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


# --- sustained wide/fear mode: hold, suppress blinking, relief burst on release
def test_wide_hold_suppresses_idle_blink_then_release_bursts(eyes_module):
    """enter_wide_hold holds the eyes wide with idle blinking fully suppressed;
    release_wide_hold ends it with a burst of relief blinks."""
    eyes_mod, left, right = eyes_module
    # Fast idle (0.05 s) would flick ~8x during the hold, absent suppression.
    e = eyes_mod.Eyes(blink_duration=0.02, min_interval=0.05, max_interval=0.05,
                      double_blink_prob=0.0, double_gap=0.05, relief_blinks=3)
    try:
        time.sleep(0.05)
        t_hold = time.monotonic()
        e.enter_wide_hold(timeout=5.0)
        time.sleep(0.4)
        # Frightened things don't blink: zero dark flicks across the whole hold.
        assert _darks_between(left, t_hold, time.monotonic()) == 0
        assert left.value is True
        assert e.is_wide_held() is True

        t_rel = time.monotonic()
        e.release_wide_hold()
        time.sleep(0.35)
        # Release produces a burst of relief blinks (not a single flick).
        assert _darks_between(left, t_rel, t_rel + 0.35) >= 2
        assert e.is_wide_held() is False
    finally:
        e.stop()


def test_wide_hold_safety_timeout_auto_releases(eyes_module):
    """A held wide/fear state must self-release after its safety timeout so a
    cancelled clip (whose release event never fires) cannot strand the eyes."""
    eyes_mod, left, right = eyes_module
    e = eyes_mod.Eyes(blink_duration=0.02, min_interval=2.0, max_interval=2.0,
                      double_blink_prob=0.0, double_gap=0.05, relief_blinks=2)
    try:
        time.sleep(0.05)
        t0 = time.monotonic()
        e.enter_wide_hold(timeout=0.2)   # backstop fires at ~0.2 s
        time.sleep(0.15)                 # still inside the hold: no blinking
        assert _darks_between(left, t0, time.monotonic()) == 0
        assert e.is_wide_held() is True
        time.sleep(0.5)                  # past the timeout -> auto-release
        assert e.is_wide_held() is False
        # Blinking resumed after the timeout (the relief burst; idle is 2 s off).
        assert _darks_between(left, t0 + 0.2, time.monotonic()) >= 1
        assert left.value is True
    finally:
        e.stop()


def test_wide_hold_suppresses_authored_blinks(eyes_module):
    """Per-frame authored eye edges must not blink through the fear hold."""
    eyes_mod, left, right = eyes_module
    e = eyes_mod.Eyes(blink_duration=0.02, min_interval=5.0, max_interval=5.0,
                      double_blink_prob=0.0)
    try:
        time.sleep(0.05)
        e.enter_wide_hold(timeout=5.0)
        t0 = time.monotonic()
        for _ in range(3):               # three authored 1->0 blink edges
            e.note_authored(0)
            e.note_authored(1)
            time.sleep(0.03)
        time.sleep(0.1)
        assert _darks_between(left, t0, time.monotonic()) == 0
        assert left.value is True
    finally:
        e.stop()


def test_slow_blink_dwell_longer_than_idle_flick(eyes_module):
    """slow_blink is one long heavy lid close/open: its dark dwell is far longer
    than the crisp idle flick (the only honest way to say 'sleepy' on on/off
    LEDs is via timing)."""
    eyes_mod, left, right = eyes_module
    e = eyes_mod.Eyes(blink_duration=0.02, min_interval=5.0, max_interval=5.0,
                      double_blink_prob=0.0)
    try:
        time.sleep(0.05)
        e.slow_blink(close=0.3)
        time.sleep(0.45)
        assert _max_dark_dwell(left) >= 0.2      # >> the 0.02 s idle flick
        assert left.value is True
    finally:
        e.stop()


def test_release_without_hold_is_noop(eyes_module):
    """release_wide_hold with no active hold must not fire a spurious burst."""
    eyes_mod, left, right = eyes_module
    e = eyes_mod.Eyes(blink_duration=0.02, min_interval=5.0, max_interval=5.0,
                      double_blink_prob=0.0)
    try:
        time.sleep(0.05)
        t0 = time.monotonic()
        e.release_wide_hold()
        time.sleep(0.15)
        assert _darks_between(left, t0, time.monotonic()) == 0
        assert e.is_wide_held() is False
    finally:
        e.stop()
