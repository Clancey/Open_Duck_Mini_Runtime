"""Pytest bootstrap for the runtime-side animation tests.

Adds the inner ``mini_bdx_runtime`` package dir to ``sys.path`` (matching
``setup.cfg`` ``package_dir = =mini_bdx_runtime``) and locates the
``open_duck_anim`` core so the tests run without installing either package. The
core is found via ``OPEN_DUCK_ANIM_HOME`` or a conventional sibling checkout.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNTIME_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_PKG_ROOT = os.path.join(_RUNTIME_ROOT, "mini_bdx_runtime")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


def _find_open_duck_anim():
    try:
        import open_duck_anim  # noqa: F401
        return
    except ImportError:
        pass
    cands = []
    env = os.environ.get("OPEN_DUCK_ANIM_HOME")
    if env:
        cands.append(env)
    siblings = os.path.dirname(_RUNTIME_ROOT)
    for name in ("Open_Duck_Mini", "clancey-didactic-memory", "open_duck_mini"):
        cands.append(os.path.join(siblings, name))
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "open_duck_anim")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return


_find_open_duck_anim()
