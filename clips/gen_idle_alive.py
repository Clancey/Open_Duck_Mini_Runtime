"""Generate ``idle_alive.duckanim`` — a hand-authored idle "alive" head loop.

Phase 4 example clip: a gentle breathing / scanning head motion with antenna
wiggle and an occasional eye blink, kept well within the DERATED head safety
envelope so it is safe for first hardware trials. Runs in any mode (it only
moves the head + antennas; legs are held neutral).

Everything is authored as smooth sinusoids whose periods divide the clip
duration, so the loop wraps seamlessly (``loop_mode="wrap"``). Head channels are
ABSOLUTE joint angles in radians (JOINT_ORDER_16 indices 5..8: neck_pitch,
head_pitch, head_yaw, head_roll); antenna tracks are normalised ``[-1, 1]`` and
live only in ``show_functions`` (never in the joint array).

Run::

    OPEN_DUCK_ANIM_HOME=/path/to/main_repo python clips/gen_idle_alive.py

It writes ``idle_alive.duckanim`` next to this file and validates it by loading
through ``open_duck_anim.load_clip`` (raises on any schema/safety violation).
"""

import json
import os
import sys

import numpy as np

# Make open_duck_anim importable (installed, env var, or sibling checkout).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV = os.environ.get("OPEN_DUCK_ANIM_HOME")
if _ENV and os.path.isdir(os.path.join(_ENV, "open_duck_anim")):
    sys.path.insert(0, _ENV)

from open_duck_anim import JOINT_ORDER_16, load_clip, clip_from_dict  # noqa: E402
from open_duck_anim import DEFAULT_ENVELOPE  # noqa: E402


FPS = 50
DURATION_S = 4.0
FRAME_COUNT = int(round(FPS * DURATION_S))   # 200 frames

# Head-channel indices within JOINT_ORDER_16.
NECK_PITCH, HEAD_PITCH, HEAD_YAW, HEAD_ROLL = 5, 6, 7, 8


def build_dict():
    n = FRAME_COUNT
    t = np.arange(n) / FPS
    w = 2.0 * np.pi / DURATION_S   # one fundamental cycle per clip -> seamless wrap

    frames = np.zeros((n, 16), dtype=np.float64)

    # Gentle "breathing" on neck_pitch (one slow cycle), subtle nod on head_pitch
    # (two cycles), a slow left-right scan on head_yaw (one cycle), and a tiny
    # roll sway. Amplitudes are deliberately small -> inside the derated envelope.
    frames[:, NECK_PITCH] = 0.045 * np.sin(w * t)                 # breathing
    frames[:, HEAD_PITCH] = 0.030 * np.sin(2.0 * w * t)           # subtle nod
    frames[:, HEAD_YAW] = 0.220 * np.sin(w * t)                   # slow scan
    frames[:, HEAD_ROLL] = 0.020 * np.sin(w * t + np.pi / 3.0)    # tiny sway

    # Antennas: gentle out-of-phase wiggle in normalised [-1, 1].
    antenna_left = 0.30 * np.sin(w * t + 0.0)
    antenna_right = 0.30 * np.sin(w * t + np.pi / 2.0)

    # Eyes: on (1) almost always; a short two-frame blink once per loop.
    eyes = np.ones(n, dtype=np.int64)
    eyes[100:102] = 0

    # One soft chirp per loop pass to exercise the sound event track.
    events = [{"frame": 8, "type": "sound", "value": "idle_chirp"}]

    return {
        "format": "duckanim",
        "version": 1,
        "name": "idle_alive",
        "fps": FPS,
        "frame_count": FRAME_COUNT,
        "duration_s": FRAME_COUNT / FPS,
        "loop_mode": "wrap",
        "layer_mask": "head",
        "requires_mode": "any",
        "priority": 0,
        "blend_in_s": 0.0,
        "blend_out_s": 0.0,
        "show_blend_in_s": 0.0,
        "show_blend_out_s": 0.0,
        "provenance": {
            "source_sha256": "hand-authored",
            "source_blend": "none (procedural, gen_idle_alive.py)",
            "source_frame_range": [0, FRAME_COUNT],
            "compiler_version": "hand-authored-1",
        },
        "joints": {
            "order": list(JOINT_ORDER_16),
            "frames": frames.tolist(),
        },
        "show_functions": {
            "antenna_left": antenna_left.tolist(),
            "antenna_right": antenna_right.tolist(),
            "eyes": eyes.tolist(),
            "events": events,
        },
        "antenna_calibration": {
            "left": {"sign": 1, "rad_min": -1.0, "rad_max": 1.0},
            "right": {"sign": -1, "rad_min": -1.0, "rad_max": 1.0},
        },
    }


def _check_derated_envelope(clip):
    """Sanity-check that the authored head motion stays inside the derated
    envelope's per-channel deflection limits and L2 budget (advisory)."""
    env = DEFAULT_ENVELOPE.derated()
    head = clip.joints[:, 5:9]                      # absolute == offset (nominal 0)
    low = np.asarray(env.low)
    high = np.asarray(env.high)
    within = np.all(head >= low - 1e-9) and np.all(head <= high + 1e-9)
    # Combined budget is a normalised norm: sqrt(sum((c/L)^2)), L=min(|low|,high).
    L = np.minimum(np.abs(low), high)
    L = np.where(L > 1e-12, L, 1e-12)
    norm = np.max(np.sqrt(np.sum((head / L) ** 2, axis=1)))
    print("  derated per-channel limits: %s .. %s" % (np.round(low, 3), np.round(high, 3)))
    print("  authored head peak normalised norm = %.3f (budget %.3f) within=%s"
          % (norm, env.l2_budget, bool(within)))


def main():
    d = build_dict()
    out_path = os.path.join(_HERE, "idle_alive.duckanim")
    with open(out_path, "w") as f:
        json.dump(d, f, indent=2)
    # Validate by loading (raises ClipValidationError on any violation).
    clip = load_clip(out_path)
    print("wrote and validated:", out_path)
    print("  frames=%d fps=%d duration=%.3fs layer_mask=%s requires_mode=%s"
          % (clip.frame_count, clip.fps, clip.duration_s, clip.layer_mask,
             clip.requires_mode))
    _check_derated_envelope(clip)
    return out_path


if __name__ == "__main__":
    main()
