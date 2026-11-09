"""Select OpenES skill backend (Classic vs full Craftax)."""

from __future__ import annotations

from training.skill_variant import VARIANT_CLASSIC, VARIANT_CRAFTAX, normalize_variant


def get_backend(variant: str = VARIANT_CLASSIC):
    v = normalize_variant(variant)
    if v == VARIANT_CRAFTAX:
        from training.skill_backends import craftax as backend

        return backend
    from training.skill_backends import classic as backend

    return backend
