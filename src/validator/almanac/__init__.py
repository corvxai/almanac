"""Almanac mechanism (vendored from sportstensor/sn41).

Self-contained subpackage. Everything except ``loop.py`` and this module is
verbatim from sportstensor/sn41 with only relative-import fixes. The public
scoring entry point ``score_almanac`` is re-exported from ``loop.py``.
"""

from .loop import score_almanac

__all__ = ["score_almanac"]
