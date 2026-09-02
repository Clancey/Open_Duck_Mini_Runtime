"""Runtime-side animation integration for Open Duck Mini v2 (Phase 4).

This subpackage wires the portable, already-tested ``open_duck_anim`` core
(the three-layer Engine, the measured head safety envelope, clip IO, the
pose->command transform and the limiters) into the on-robot 50 Hz control
loop. It adds ONLY the runtime concerns the core deliberately leaves out:

* a **mode FSM** with safe startup (``BOOT/DISARMED`` -> ``ARMING``), the
  operational modes (``DOCK_DEMO`` / ``STAND`` / ``WALK``) and a latched
  ``FAULT`` (see :mod:`.fsm`);
* the **minimum safety set** — controlled abort, e-stop/deadman, watchdogs,
  thermal/load management (see :mod:`.safety`);
* a **mockable hardware abstraction** so the FSM and safety logic are testable
  with no robot attached (see :mod:`.hardware`);
* the **integration controller** that evaluates the engine exactly once per
  control tick from a single monotonic clock, routes the head output through
  the additive path + safe envelope, applies the capability matrix and the
  final 14-DOF bus velocity clip (see :mod:`.controller`).

``open_duck_anim`` must be importable. On the Pi it is ``pip install``-ed. For
local development / CI where it lives in a sibling checkout, set the
``OPEN_DUCK_ANIM_HOME`` environment variable to the repo root (the directory
that contains the ``open_duck_anim/`` package) and importing this subpackage
will add it to ``sys.path``.
"""

import os
import sys


def _ensure_open_duck_anim():
    """Make ``open_duck_anim`` importable (installed, env var, or sibling).

    Order: (1) already importable -> nothing to do; (2) ``OPEN_DUCK_ANIM_HOME``
    env var; (3) a couple of conventional sibling-checkout locations relative to
    this runtime repo. Never raises here — a genuine ``ImportError`` surfaces at
    the real ``import open_duck_anim`` site with a clear message.
    """
    try:
        import open_duck_anim  # noqa: F401
        return
    except ImportError:
        pass

    candidates = []
    env = os.environ.get("OPEN_DUCK_ANIM_HOME")
    if env:
        candidates.append(env)
    # Conventional sibling checkouts (…/Open_Duck_Mini_Runtime -> siblings).
    here = os.path.dirname(os.path.abspath(__file__))
    runtime_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    siblings_root = os.path.dirname(runtime_root)
    for name in ("Open_Duck_Mini", "open_duck_mini", "clancey-didactic-memory"):
        candidates.append(os.path.join(siblings_root, name))

    for cand in candidates:
        if cand and os.path.isdir(os.path.join(cand, "open_duck_anim")):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return


_ensure_open_duck_anim()

from .hardware import (  # noqa: E402
    RobotInterface,
    SensorSnapshot,
    OperatorInput,
    MockRobot,
    N_DOFS,
    HEAD_HW_SLICE,
    LEG_HW_INDICES,
)
from .fsm import (  # noqa: E402
    ModeFSM,
    FSMState,
    FSMConfig,
    GuardBand,
)
from .safety import (  # noqa: E402
    SafetyMonitor,
    SafetyConfig,
    FaultAction,
    DeadlineWatchdog,
    StaleCommandWatchdog,
    ThermalManager,
    SafetyStatus,
)
from .controller import (  # noqa: E402
    AnimationController,
    ControllerConfig,
    ControllerOutput,
    make_head_engine,
)

__all__ = [
    "RobotInterface",
    "SensorSnapshot",
    "OperatorInput",
    "MockRobot",
    "N_DOFS",
    "HEAD_HW_SLICE",
    "LEG_HW_INDICES",
    "ModeFSM",
    "FSMState",
    "FSMConfig",
    "GuardBand",
    "SafetyMonitor",
    "SafetyConfig",
    "FaultAction",
    "DeadlineWatchdog",
    "StaleCommandWatchdog",
    "ThermalManager",
    "SafetyStatus",
    "AnimationController",
    "ControllerConfig",
    "ControllerOutput",
    "make_head_engine",
]
