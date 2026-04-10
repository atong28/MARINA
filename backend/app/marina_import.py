"""
Bootstrap module: inserts the MARINA project root into sys.path so that
`src.modules.*` can be imported from the main repo without cloning a copy.

Call `ensure_marina_importable()` once at startup (or rely on module-level
execution when this file is first imported).
"""
import sys
import os
import logging

logger = logging.getLogger(__name__)

_bootstrapped = False


def ensure_marina_importable() -> None:
    """Add MARINA_ROOT to sys.path if not already present."""
    global _bootstrapped
    if _bootstrapped:
        return

    from app.config import MARINA_ROOT

    if not os.path.isdir(MARINA_ROOT):
        raise RuntimeError(
            f"MARINA_ROOT={MARINA_ROOT!r} is not a directory. "
            "Set the MARINA_ROOT environment variable to the path of the MARINA project."
        )

    src_dir = os.path.join(MARINA_ROOT, "src")
    if not os.path.isdir(src_dir):
        raise RuntimeError(
            f"Expected {src_dir!r} to exist. "
            "MARINA_ROOT must be the directory that contains the 'src/' package."
        )

    if MARINA_ROOT not in sys.path:
        sys.path.insert(0, MARINA_ROOT)
        logger.info("Added MARINA_ROOT=%s to sys.path", MARINA_ROOT)

    _bootstrapped = True
