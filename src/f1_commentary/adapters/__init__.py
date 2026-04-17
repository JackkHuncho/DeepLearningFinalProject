"""Adapter / bridge layer between even-phase and odd-phase F1 commentary schemas.

Even phases live in ``src/f1_commentary/`` and odd phases live in
``code/src/telemetry_to_narrative/``.  The functions exported here translate
data models in both directions so the two codebases can interoperate without
either side needing to import the other's schemas directly.

.. note::

   The odd-phase package (``telemetry_to_narrative``) expects
   ``code/`` to be on ``PYTHONPATH``.  The bridge module adds it to
   ``sys.path`` automatically at import time, but if you need to use the
   odd-phase package elsewhere, ensure ``code/`` is on the path.
"""

from f1_commentary.adapters.bridge import (
    EVEN_TO_ODD_EVENT_TYPE,
    ODD_TO_EVEN_EVENT_TYPE,
    even_candidate_to_odd,
    even_guarded_to_odd_generated,
    odd_beat_to_even_candidate,
    odd_commentary_to_even,
    odd_final_to_even,
    odd_snapshot_to_even,
    odd_telemetry_to_even,
)

__all__ = [
    "EVEN_TO_ODD_EVENT_TYPE",
    "ODD_TO_EVEN_EVENT_TYPE",
    "even_candidate_to_odd",
    "even_guarded_to_odd_generated",
    "odd_beat_to_even_candidate",
    "odd_commentary_to_even",
    "odd_final_to_even",
    "odd_snapshot_to_even",
    "odd_telemetry_to_even",
]
