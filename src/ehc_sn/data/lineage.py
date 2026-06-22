"""Generic artifact-lineage validation for derived task corpora.

Provides :func:`validate_shared_parent` to check that two task corpora
reference the same parent artifact for a given role.  This enables
cross-task transfer claims (e.g. Goaltrace ↔ Routebind sharing the same
semantic DAG).
"""

from __future__ import annotations

from ehc_sn.data.manifest import normalize_manifest_parents


# =============================================================================
def validate_shared_parent(
    manifest_a: dict,
    manifest_b: dict,
    role: str,
    *,
    strict_digest: bool = True,
) -> None:
    """Validate that two task corpora share the same parent artifact for a
    given role.

    Content digest comparison is authoritative.  Artifact ID comparison
    provides human-readable correspondence.  When *strict_digest* is
    ``True`` (default), both digests must be present and equal.

    Args:
        manifest_a: Parsed manifest for the first task corpus.
        manifest_b: Parsed manifest for the second task corpus.
        role: Parent role name (e.g. ``"semantic_graph"``).
        strict_digest: When ``True``, require both manifests to have
            ``content_digest`` and assert equality.

    Raises:
        ValueError: On any mismatch or missing parent reference.
    """
    parents_a = normalize_manifest_parents(manifest_a)
    parents_b = normalize_manifest_parents(manifest_b)

    if role not in parents_a:
        raise ValueError(
            f"Manifest A has no parent with role {role!r}. "
            f"Available roles: {sorted(parents_a)}"
        )
    if role not in parents_b:
        raise ValueError(
            f"Manifest B has no parent with role {role!r}. "
            f"Available roles: {sorted(parents_b)}"
        )

    ref_a = parents_a[role]
    ref_b = parents_b[role]

    # Family must match
    fam_a = ref_a.get("family")
    fam_b = ref_b.get("family")
    if fam_a and fam_b and fam_a != fam_b:
        raise ValueError(
            f"Family mismatch for role {role!r}: " f"{fam_a!r} != {fam_b!r}."
        )

    # Content digest is authoritative
    dig_a = ref_a.get("content_digest")
    dig_b = ref_b.get("content_digest")

    if strict_digest and (dig_a is None or dig_b is None):
        raise ValueError(
            f"Cannot validate shared parent for role {role!r}: "
            f"content_digest missing (A={dig_a!r}, B={dig_b!r})."
        )

    if dig_a is not None and dig_b is not None and dig_a != dig_b:
        raise ValueError(
            f"Content digest mismatch for role {role!r}: "
            f"{dig_a} != {dig_b}."
        )

    # Artifact ID provides human-readable correspondence
    id_a = ref_a.get("artifact_id")
    id_b = ref_b.get("artifact_id")
    if id_a is not None and id_b is not None and id_a != id_b:
        raise ValueError(
            f"Artifact ID mismatch for role {role!r}: " f"{id_a!r} != {id_b!r}."
        )


__all__ = ["validate_shared_parent"]
