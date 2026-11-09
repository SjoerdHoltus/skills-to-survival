"""Craftax variant labels and run folder names for OpenES primitive skills."""

from __future__ import annotations

VARIANT_CLASSIC = "classic"
VARIANT_CRAFTAX = "craftax"
VALID_VARIANTS = (VARIANT_CLASSIC, VARIANT_CRAFTAX)


def normalize_variant(variant: str) -> str:
    v = (variant or VARIANT_CLASSIC).strip().lower()
    if v in ("full", "full_craftax", "craftax_full"):
        v = VARIANT_CRAFTAX
    if v not in VALID_VARIANTS:
        raise ValueError(f"variant must be one of {VALID_VARIANTS}, got {variant!r}")
    return v


def runs_policy_folder(skill: str, variant: str) -> str:
    """e.g. ``find_wood_classic`` / ``find_wood_craftax`` under ``runs/``."""
    skill_key = (skill or "").strip().lower()
    if not skill_key:
        raise ValueError("skill name required")
    return f"{skill_key}_{normalize_variant(variant)}"
