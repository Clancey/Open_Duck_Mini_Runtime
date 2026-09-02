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
  ``hold_open(seconds)`` (suppress blinking and stay wide, e.g. "startle"/"wide").
  The per-frame authored eye channel is honoured via :meth:`note_authored` (a
  1->0 edge = an authored blink).

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
                 double_blink_prob=0.18, double_gap=0.16):
        self.left_eye = digitalio.DigitalInOut(LEFT_EYE_PIN)
        self.left_eye.direction = digitalio.Direction.OUTPUT
        self.right_eye = digitalio.DigitalInOut(RIGHT_EYE_PIN)
        self.right_eye.direction = digitalio.Direction.OUTPUT

        self.blink_duration = blink_duration
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.double_blink_prob = double_blink_prob
        self.double_gap = double_gap

        self._lock = Lock()
        self._blink_requests = 0        # pending explicit blinks (expressive cues)
        self._hold_until = 0.0          # monotonic time to suppress auto-blink until
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

    # -- thread-safe public cues (called from the control loop) --------------
    def blink(self, n=1):
        """Request ``n`` explicit blink(s) as soon as possible."""
        with self._lock:
            self._blink_requests += max(1, int(n))

    def double_blink(self):
        """A quick two-blink flutter (e.g. a 'happy' cue)."""
        with self._lock:
            self._blink_requests += 2

    def hold_open(self, seconds=1.0):
        """Suppress idle blinking and keep the eyes wide for ``seconds`` (e.g.
        'startle'/'wide')."""
        with self._lock:
            self._hold_until = max(self._hold_until, time.monotonic() + float(seconds))
            self._blink_requests = 0
        self._set_eyes(True)

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
                    req = self._blink_requests
                    self._blink_requests = 0
                    holding = now < self._hold_until
                if holding:
                    if not self._lit:
                        self._set_eyes(True)
                elif req > 0:
                    self._do_blink()
                    if req >= 2:
                        time.sleep(self.double_gap)
                        self._do_blink()
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
