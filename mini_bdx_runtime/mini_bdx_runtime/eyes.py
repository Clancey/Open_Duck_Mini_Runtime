"""Eye lights for Open Duck Mini v2.

Two digital LED channels (``board.D24`` left, ``board.D23`` right). The eyes are
lit (open) as the resting state and *blink* by going briefly dark. A background
thread produces a natural, irregular idle blink; expressive clip cues override it.

Design (matches the three-layer engine's background/triggered model):

* **Background idle blink** — the resting behaviour. Blink interval is random in
  ``[min_interval, max_interval]`` (default 2-6 s) so it never looks metronomic,
  with an occasional spontaneous double-blink. This runs whenever the eyes exist.
* **Triggered overrides take precedence** — a clip eye event nudges the eyes:
  ``blink`` (one blink now), ``double_blink`` (two quick blinks, e.g. "happy"),
  ``hold_open(seconds)`` (suppress blinking and stay wide *briefly*, e.g. a
  one-shot "startle"), ``slow_blink`` (one long, heavy lid close/open, e.g.
  "sleepy"/"content"). The per-frame authored eye channel is honoured via
  :meth:`note_authored` (a 1->0 edge = an authored blink).
* **Sustained wide/fear mode** — :meth:`enter_wide_hold` holds the eyes wide
  and *suppresses idle blinking indefinitely* (frightened things don't blink),
  until :meth:`release_wide_hold` fires a burst of relief blinks. Unlike the
  fire-and-forget cues this is a *state* that must be released; a safety timeout
  (``wide_hold_timeout``, default 8 s) is a backstop so a cancelled clip can
  never leave the eyes stuck wide forever — the timeout auto-releases with the
  same relief burst.

Hardware note: these are two *binary* on/off LED channels (no PWM, brightness,
RGB or eyelid radius), so "wide" and "slow blink" are expressed purely through
*timing and blink behaviour*, never aperture. A "squint" (partial aperture)
cannot be represented on this hardware and is deliberately not faked here.

ALL pin writes happen on the single background thread, so external callers only
set thread-safe flags — there is never a cross-thread race on the LEDs.
"""

import board
import digitalio
import random
import time
from threading import Thread, Event, Lock

LEFT_EYE_PIN = board.D24
RIGHT_EYE_PIN = board.D23


class Eyes:
    def __init__(self, blink_duration=0.12, min_interval=2.0, max_interval=6.0,
                 double_blink_prob=0.18, double_gap=0.16, slow_blink_duration=0.55,
                 wide_hold_timeout=8.0, relief_blinks=3, max_burst=3):
        self.left_eye = digitalio.DigitalInOut(LEFT_EYE_PIN)
        self.left_eye.direction = digitalio.Direction.OUTPUT
        self.right_eye = digitalio.DigitalInOut(RIGHT_EYE_PIN)
        self.right_eye.direction = digitalio.Direction.OUTPUT

        self.blink_duration = blink_duration
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.double_blink_prob = double_blink_prob
        self.double_gap = double_gap
        self.slow_blink_duration = float(slow_blink_duration)
        # Safety backstop: the sustained wide/fear state auto-releases after at
        # most this many seconds, so a cancelled clip (whose release event never
        # fires) can never leave the eyes stuck wide forever.
        self.wide_hold_timeout = float(wide_hold_timeout)
        self.relief_blinks = int(relief_blinks)
        self.max_burst = int(max_burst)

        self._lock = Lock()
        self._blink_requests = 0        # pending explicit blinks (expressive cues)
        self._slow_requests = 0         # pending slow (heavy) blinks
        self._slow_close = self.slow_blink_duration
        self._hold_until = 0.0          # monotonic time to suppress auto-blink until
        self._fear_active = False       # sustained wide/fear state (blink-suppressed)
        self._fear_until = 0.0          # safety-timeout deadline for the fear state
        self._authored_prev = 1         # last per-frame authored eye state (1=open)
        self._lit = False

        self._stop_event = Event()
        self._set_eyes(True)            # resting state: eyes open/lit
        self._next_auto = time.monotonic() + self._rand_interval()
        self._thread = Thread(target=self.run, daemon=True)
        self._thread.start()

    # -- low level -----------------------------------------------------------
    def _rand_interval(self):
        return random.uniform(self.min_interval, self.max_interval)

    def _set_eyes(self, state):
        s = bool(state)
        self.left_eye.value = s
        self.right_eye.value = s
        self._lit = s

    def _do_blink(self):
        self._set_eyes(False)
        time.sleep(self.blink_duration)
        self._set_eyes(True)

    def _do_burst(self, n):
        """Blink ``n`` times back to back (capped at ``max_burst``), e.g. the
        relief flurry when the fear state is released."""
        count = max(1, min(int(n), self.max_burst))
        for i in range(count):
            if i:
                time.sleep(self.double_gap)
            self._do_blink()

    def _do_slow_blink(self, close):
        """A single long, heavy lid close/open. On binary LEDs this is just a
        longer dark dwell than the crisp idle flick — the 'sleepy'/'content'
        feel comes entirely from the timing."""
        self._set_eyes(False)
        time.sleep(max(0.0, float(close)))
        self._set_eyes(True)

    # -- thread-safe public cues (called from the control loop) --------------
    def blink(self, n=1):
        """Request ``n`` explicit blink(s) as soon as possible."""
        with self._lock:
            self._blink_requests += max(1, int(n))

    def double_blink(self):
        """A quick two-blink flutter (e.g. a 'happy' cue)."""
        with self._lock:
            self._blink_requests += 2

    def slow_blink(self, n=1, close=None):
        """One (or ``n``) long, heavy lid close/open (e.g. 'sleepy'/'content'/
        'sad'). Suppressed while a wide/fear hold is active."""
        with self._lock:
            self._slow_requests += max(1, int(n))
            if close is not None:
                self._slow_close = float(close)

    def hold_open(self, seconds=1.0):
        """Suppress idle blinking and keep the eyes wide for ``seconds`` (a
        one-shot 'startle'). For a sustained, explicitly-released fear state use
        :meth:`enter_wide_hold`."""
        with self._lock:
            self._hold_until = max(self._hold_until, time.monotonic() + float(seconds))
            self._blink_requests = 0
        self._set_eyes(True)

    def enter_wide_hold(self, timeout=None):
        """Enter the sustained wide/fear state: eyes held lit and idle blinking
        suppressed *indefinitely* until :meth:`release_wide_hold`. A safety
        timeout (``wide_hold_timeout`` unless overridden) auto-releases the state
        as a backstop, so a cancelled/aborted clip can never leave the eyes
        stuck wide."""
        t = self.wide_hold_timeout if timeout is None else float(timeout)
        with self._lock:
            self._fear_active = True
            self._fear_until = time.monotonic() + t
            self._blink_requests = 0
            self._slow_requests = 0
        self._set_eyes(True)

    def release_wide_hold(self, relief_blinks=None):
        """Release the sustained wide/fear state. If it was active, emit a burst
        of relief blinks (``relief_blinks``; pass 0 for a silent release, as on
        FAULT/e-stop). Idempotent — a no-op if no hold is active."""
        n = self.relief_blinks if relief_blinks is None else max(0, int(relief_blinks))
        with self._lock:
            was = self._fear_active
            self._fear_active = False
            self._fear_until = 0.0
            if was:
                self._blink_requests += n
                self._next_auto = time.monotonic() + self._rand_interval()

    def is_wide_held(self):
        """True while the sustained wide/fear state is active (test/telemetry)."""
        with self._lock:
            return self._fear_active

    def note_authored(self, state):
        """Honour a clip's per-frame eye channel: a 1->0 edge is an authored
        blink. Baseline lighting is owned by the background loop, so a steady
        value (e.g. a clip with no eye channel) does not force the eyes dark."""
        s = int(bool(state))
        with self._lock:
            edge = (self._authored_prev == 1 and s == 0)
            self._authored_prev = s
            if edge:
                self._blink_requests += 1

    def set_manual(self, state):
        """Explicit momentary hold of a raw eye state (rarely needed)."""
        with self._lock:
            self._hold_until = time.monotonic() + 0.25
        self._set_eyes(bool(state))

    # -- background loop -----------------------------------------------------
    def run(self):
        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                with self._lock:
                    # Safety backstop: auto-release a held wide/fear state once
                    # its timeout elapses, emitting the same relief burst, so a
                    # lost release event can never strand the eyes wide.
                    if self._fear_active and now >= self._fear_until:
                        self._fear_active = False
                        self._fear_until = 0.0
                        self._blink_requests += self.relief_blinks
                        self._next_auto = now + self._rand_interval()
                    fear = self._fear_active
                    req = self._blink_requests
                    self._blink_requests = 0
                    slow = self._slow_requests
                    self._slow_requests = 0
                    slow_close = self._slow_close
                    holding = fear or now < self._hold_until
                if holding:
                    # Wide/fear or a one-shot hold: stay lit, drop queued blinks.
                    if not self._lit:
                        self._set_eyes(True)
                elif slow > 0:
                    self._do_slow_blink(slow_close)
                    with self._lock:
                        self._next_auto = time.monotonic() + self._rand_interval()
                elif req > 0:
                    self._do_burst(req)
                    with self._lock:
                        self._next_auto = time.monotonic() + self._rand_interval()
                elif now >= self._next_auto:
                    self._do_blink()
                    if random.random() < self.double_blink_prob:
                        time.sleep(self.double_gap)
                        self._do_blink()
                    with self._lock:
                        self._next_auto = time.monotonic() + self._rand_interval()
                else:
                    if not self._lit:
                        self._set_eyes(True)
                self._stop_event.wait(0.02)
        except Exception as err:
            print(f"Error in eye thread: {err}")
            self._stop_event.set()

    def stop(self):
        self._stop_event.set()
        with self._lock:
            self._fear_active = False
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._set_eyes(False)
            self.left_eye.deinit()
            self.right_eye.deinit()
        except Exception:
            pass


if __name__ == "__main__":
    e = Eyes()
    try:
        while True:
            time.sleep(1)
    finally:
        e.stop()
