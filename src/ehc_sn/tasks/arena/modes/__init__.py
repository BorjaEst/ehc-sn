"""Arena mode-specific capability bindings.

Sub-modules:

- :mod:`ehc_sn.tasks.arena.modes.replay` — teacher-forced replay execution-mode
  binding (runtime, trajectory helpers, batch-key contracts).
"""

from ehc_sn.tasks.arena.modes import replay

__all__ = ["replay"]
