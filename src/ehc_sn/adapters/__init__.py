"""
Models should be task agnostic, this module provides a stable interface for all
models to be used by the rest of the codebase.

This is where we should put any adapter code that is specific to a particular
task, as MazeHard, Dungeons, or any other environment we want to test a model
on. The idea is to keep the core EHC model code clean and focused on theory,
and put any task-specific logic in these adapters.

"""
