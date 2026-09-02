"""Dock-demo entry point — play a ``.duckanim`` library in DOCK_DEMO, no RL.

This is the fastest path to a visible result on the physical robot: it bypasses
the policy entirely (plan §6.2 DOCK_DEMO), holds the legs at ``init_pos`` and
animates the head / neck / antennas / eyes / sounds at 50 Hz through the full
integrated safety stack (mode FSM + minimum safety set + derated head envelope +
final bus velocity clip).

Mode sequence enforced by the FSM (plan §4.3):

    BOOT/DISARMED --(arm)--> ARMING --(ramp done + DOCK confirm)--> DOCK_DEMO

so the robot NEVER snaps: ARMING captures the measured pose and ramps torque and
targets to ``init_pos`` before any animation plays. DOCK_DEMO requires an
explicit dock confirmation to enter.

Run on the robot::

    OPEN_DUCK_ANIM_HOME=/path/to/main_repo \\
      python scripts/dock_demo.py \\
        --clip clips/idle_alive.duckanim \\
        --clip-lib clips \\
        --sound-dir /path/to/wavs

Controls (interactive TTY):  a=arm  d=confirm-dock  1..9=fire clip  x=e-stop
                             r=reset-fault  q=quit.  ``--auto`` arms + docks
automatically after a short countdown (hands-off demo).

SAFETY: keep a hand near the physical power switch. E-stop (``x`` or ``--auto``
watchdog) latches FAULT; the default fault action is CONTROLLED_HOLD (the robot
holds its last pose rather than going limp — safer for a small standing biped).
"""

import argparse
import glob
import os
import sys
import time

import numpy as np

# --- make the runtime package and open_duck_anim importable -------------------
_THIS = os.path.dirname(os.path.abspath(__file__))
_RUNTIME_ROOT = os.path.abspath(os.path.join(_THIS, ".."))
# The importable package root is the inner ``mini_bdx_runtime`` dir (setup.cfg:
# ``package_dir = =mini_bdx_runtime``). Add ONLY that so ``import
# mini_bdx_runtime`` resolves to the real package, not the empty outer shim.
_PKG_ROOT = os.path.join(_RUNTIME_ROOT, "mini_bdx_runtime")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from mini_bdx_runtime.anim import (  # noqa: E402
    AnimationController, ControllerConfig, OperatorInput, FSMState,
)
from mini_bdx_runtime.anim import _ensure_open_duck_anim  # noqa: E402

_ensure_open_duck_anim()
from open_duck_anim import load_clip, Triggers  # noqa: E402


CTRL_DT = 0.02  # 50 Hz


class _KeyReader:
    """Non-blocking single-char stdin reader (raw TTY). No-op if not a TTY."""

    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self._fd = None
        self._old = None

    def __enter__(self):
        if not self.enabled:
            return self
        import termios, tty
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *exc):
        if self.enabled and self._old is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def read(self):
        if not self.enabled:
            return None
        import select
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            return sys.stdin.read(1)
        return None


def _load_library(clip_lib):
    lib = {}
    if not clip_lib:
        return lib
    for path in sorted(glob.glob(os.path.join(clip_lib, "*.duckanim"))):
        try:
            clip = load_clip(path)
            lib[clip.name] = clip
        except Exception as e:  # a bad clip must not crash the demo
            print("  skip %s: %s" % (os.path.basename(path), e))
    return lib


def main():
    ap = argparse.ArgumentParser(description="Open Duck Mini v2 dock demo (no RL)")
    ap.add_argument("--clip", default=os.path.join(_RUNTIME_ROOT, "clips", "idle_alive.duckanim"),
                    help="background idle clip (.duckanim)")
    ap.add_argument("--clip-lib", default=os.path.join(_RUNTIME_ROOT, "clips"),
                    help="directory of triggerable .duckanim clips (keys 1..9)")
    ap.add_argument("--usb-port", default="/dev/ttyACM0")
    ap.add_argument("--sound-dir", default="./")
    ap.add_argument("--derating", type=float, default=0.5,
                    help="head envelope derating (0.5 = first-trial default; "
                         "raise toward 1.0 as hardware data accrues)")
    ap.add_argument("--max-demo-s", type=float, default=120.0,
                    help="max continuous demo duration before cooldown (thermal)")
    ap.add_argument("--auto", action="store_true",
                    help="auto arm + dock after a countdown (hands-off)")
    ap.add_argument("--no-sounds", action="store_true")
    ap.add_argument("--no-projector", action="store_true")
    ap.add_argument("--no-eyes", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="use the mock robot (no hardware) to smoke-test the loop")
    args = ap.parse_args()

    background = load_clip(args.clip)
    print("background clip:", background.name)
    library = _load_library(args.clip_lib)
    trigger_names = [n for n in library if n != background.name]
    print("trigger clips:", trigger_names or "(none)")

    # --- robot ---
    if args.dry_run:
        from mini_bdx_runtime.anim import MockRobot, SensorSnapshot

        class _DryRobot(MockRobot):
            def read(self, operator):
                # Simulate a perfect servo: measured pose follows the last
                # commanded targets (one-tick lag) so ARMING tracking passes.
                pos = self.last_targets if self.last_targets is not None else np.zeros(14)
                return SensorSnapshot(
                    t_monotonic=time.monotonic(),
                    joint_positions=np.asarray(pos, dtype=np.float64).copy(),
                    joint_velocities=np.zeros(14),
                    tilt_rad=0.0, feet_contacts=np.array([1.0, 1.0]),
                    operator=operator,
                )
        robot = _DryRobot()
    else:
        from mini_bdx_runtime.anim.real_robot import RealRobot
        robot = RealRobot(
            usb_port=args.usb_port, sound_directory=args.sound_dir,
            enable_sounds=not args.no_sounds, enable_projector=not args.no_projector,
            enable_eyes=not args.no_eyes,
        )
        robot.connect()

    from mini_bdx_runtime.anim.safety import SafetyConfig
    scfg = SafetyConfig(max_continuous_load_s=args.max_demo_s)
    ccfg = ControllerConfig(envelope_derating=args.derating)
    controller = AnimationController(
        robot, background_clip=background, config=ccfg, safety_config=scfg,
    )

    op = OperatorInput()
    auto_t0 = time.monotonic() if args.auto else None
    print("\nReady. %s\n" % ("AUTO: arming shortly..." if args.auto
                             else "Press 'a' to arm, 'd' to confirm dock."))

    overruns = 0
    tick = 0
    with _KeyReader() as keys:
        try:
            while True:
                t_start = time.monotonic()
                triggers = Triggers()

                # --- operator input (edge-triggered flags cleared each tick) ---
                op.arm_requested = False
                op.dock_confirmed = False
                op.fault_reset = False
                op.estop = op.estop  # estop is latching until reset

                if args.auto and auto_t0 is not None:
                    dt = time.monotonic() - auto_t0
                    if 1.0 < dt < 1.0 + CTRL_DT:
                        op.arm_requested = True
                    if controller.fsm.state == FSMState.ARMING:
                        op.dock_confirmed = True  # confirm as soon as armed

                k = keys.read()
                if k:
                    if k == "a":
                        op.arm_requested = True
                    elif k == "d":
                        op.dock_confirmed = True
                    elif k == "x":
                        op.estop = True
                    elif k == "r":
                        op.estop = False
                        op.fault_reset = True
                    elif k == "q":
                        break
                    elif k.isdigit() and k != "0":
                        idx = int(k) - 1
                        if idx < len(trigger_names):
                            triggers = Triggers(clips=[library[trigger_names[idx]]])
                            print("  trigger:", trigger_names[idx])

                out = controller.step(op, triggers=triggers)

                if out is not None and out.fsm.changed:
                    print("  state -> %s%s" % (
                        out.state.value,
                        (" (%s)" % out.fsm.fault_reason) if out.fsm.fault_reason else ""))
                if out is not None and out.request_cooldown and tick % 50 == 0:
                    print("  THERMAL cooldown active (duty limit) — head blending to neutral")

                # --- pace the loop to 50 Hz; count budget overruns ---
                elapsed = time.monotonic() - t_start
                if elapsed > CTRL_DT:
                    overruns += 1
                    if overruns % 50 == 1:
                        print("  Policy control budget exceeded (%.1f ms)" % (elapsed * 1e3))
                else:
                    time.sleep(CTRL_DT - elapsed)
                tick += 1
        except KeyboardInterrupt:
            pass
        finally:
            print("\nshutting down: torque off + show neutral")
            try:
                robot.shutdown_show()
                robot.torque_off()
                if hasattr(robot, "close"):
                    robot.close()
            except Exception as e:
                print("  shutdown warning:", e)
    print("done. ticks=%d overruns=%d" % (tick, overruns))


if __name__ == "__main__":
    main()
