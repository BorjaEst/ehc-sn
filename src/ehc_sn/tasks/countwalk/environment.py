"""Countwalk task live-environment shell — deferred.

This file is the canonical location for the Countwalk task-owned live
sequential-decision shell (``tasks/countwalk/environment.py``).  A live
TorchRL environment implementation for Countwalk RL rollouts has not yet been
built.  When implemented, it will wire reward, terminated, truncated, and
episode-horizon semantics here rather than in ``envs/``.

Current status: **not implemented**.  No Countwalk live-env support exists.
The V1 Countwalk training path uses the replay trajectory controller
(:class:`~ehc_sn.tasks.countwalk.capabilities.replay.CountwalkReplayCapability`)
and does not require a live stepping environment.
"""
