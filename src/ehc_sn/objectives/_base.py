# Removed: BaseObjective was a dead abstraction (2026-06-19).
# Composites (ACTSupervisedScorer, TEMObjective) are now direct nn.Module
# subclasses. Rollout traversal is owned by rollouts/scoring.py.
# No code references this module.
