"""Central optimizer acronym resolution for MEALPY and project optimizers."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import mealpy

from dbo_optimizer import DBOOptimizer
from de_awad_optimizer import DE_AWAD
from de_diversity_selection_optimizer import DE_DiversitySelection
from de_mahalanobis_optimizer import DE_Mahalanobis
from de_m_optimizer import DE_M
from de_mc_cf_optimizer import DE_MC_CF
from de_mc_optimizer import DE_MC
from dsade_awad_optimizer import DSADE_AWAD
from dsade_optimizer import DSADE
from macro_de_optimizer import MaCRO_DE


VARIANT_PREFIXES = ("Original", "Base", "Dev")
VARIANT_PRIORITY = {
    "Original": 0,
    "Base": 1,
    "Dev": 2,
}

CUSTOM_OPTIMIZER_CLASSES = {
    "MaCRO-DE": MaCRO_DE,
    "DSADE": DSADE,
    "DSADE_AWAD": DSADE_AWAD,
    "DBO": DBOOptimizer,
    "DE_AWAD": DE_AWAD,
    "DE_DiversitySelection": DE_DiversitySelection,
    "DE_Mahalanobis": DE_Mahalanobis,
    "DE-M": DE_M,
    "DE-MC": DE_MC,
    "DE-MC-CF": DE_MC_CF,
}
CUSTOM_OPTIMIZERS = tuple(CUSTOM_OPTIMIZER_CLASSES)

CUSTOM_OPTIMIZER_ALIASES = {
    "MACRO-DE": "MaCRO-DE",
    "MACRO_DE": "MaCRO-DE",
    "MACRODE": "MaCRO-DE",
    "DSADE": "DSADE",
    "DSA-DE": "DSADE",
    "DSA_DE": "DSADE",
    "DSADE_AWAD": "DSADE_AWAD",
    "DSADE-AWAD": "DSADE_AWAD",
    "DBO": "DBO",
    "DE-AWAD": "DE_AWAD",
    "DE_AWAD": "DE_AWAD",
    "DE-DIVERSITYSELECTION": "DE_DiversitySelection",
    "DE_DIVERSITYSELECTION": "DE_DiversitySelection",
    "DE-MAHALANOBIS": "DE_Mahalanobis",
    "DE_MAHALANOBIS": "DE_Mahalanobis",
    "DE-M": "DE-M",
    "DE_M": "DE-M",
    "DE-MC": "DE-MC",
    "DE_MC": "DE-MC",
    "DE-MC-CF": "DE-MC-CF",
    "DE_MC_CF": "DE-MC-CF",
}

MEALPY_OPTIMIZER_ALIASES = {
    "dmo": "DMOA",
}


def normalize_optimizer_key(name: str) -> str:
    """Normalize case, hyphens, underscores, and spacing for matching."""
    return "".join(
        char.casefold()
        for char in str(name).strip()
        if char.isalnum()
    )


@lru_cache(maxsize=1)
def _available_optimizers() -> dict[str, type]:
    return mealpy.get_all_optimizers(verbose=False)


@lru_cache(maxsize=1)
def _mealpy_name_indexes() -> tuple[dict[str, str], dict[str, str]]:
    exact_index = {}
    short_candidates = defaultdict(list)

    for optimizer_name in _available_optimizers():
        exact_index[normalize_optimizer_key(optimizer_name)] = optimizer_name

        for prefix in VARIANT_PREFIXES:
            if (
                optimizer_name.startswith(prefix)
                and len(optimizer_name) > len(prefix)
            ):
                short_name = optimizer_name[len(prefix):]
                short_candidates[normalize_optimizer_key(short_name)].append(
                    optimizer_name
                )
                break

    short_index = {
        key: _preferred_variant(candidates)
        for key, candidates in short_candidates.items()
    }

    return exact_index, short_index


def _preferred_variant(candidates: list[str]) -> str:
    return sorted(candidates, key=_variant_sort_key)[0]


def _variant_sort_key(optimizer_name: str) -> tuple[int, str]:
    for prefix in VARIANT_PREFIXES:
        if optimizer_name.startswith(prefix):
            return (
                VARIANT_PRIORITY[prefix],
                optimizer_name.casefold(),
            )
    return (
        len(VARIANT_PRIORITY),
        optimizer_name.casefold(),
    )


def _resolve_custom_optimizer(name: str) -> str | None:
    key = normalize_optimizer_key(name)
    normalized_aliases = {
        normalize_optimizer_key(alias): custom_name
        for alias, custom_name in CUSTOM_OPTIMIZER_ALIASES.items()
    }
    return normalized_aliases.get(key)


def _resolve_mealpy_optimizer(name: str) -> str | None:
    exact_index, short_index = _mealpy_name_indexes()
    key = normalize_optimizer_key(name)

    resolved_alias = MEALPY_OPTIMIZER_ALIASES.get(key)
    if resolved_alias is not None:
        return _resolve_mealpy_optimizer(resolved_alias)

    if key in exact_index:
        return exact_index[key]

    return short_index.get(key)


def resolve_optimizer_name(name: str) -> str:
    """Return the installed MEALPY class name or canonical custom name."""
    custom_name = _resolve_custom_optimizer(name)
    if custom_name is not None:
        return custom_name

    mealpy_name = _resolve_mealpy_optimizer(name)
    if mealpy_name is not None:
        return mealpy_name

    raise ValueError(
        f"Unknown optimizer '{name}'. Use a MEALPY optimizer acronym/class name "
        "or one of the project custom optimizers."
    )


def optimizer_acronym(name: str) -> str:
    """Return the short display acronym for a supported optimizer."""
    resolved_name = resolve_optimizer_name(name)
    if resolved_name in CUSTOM_OPTIMIZERS:
        return resolved_name

    for prefix in VARIANT_PREFIXES:
        if resolved_name.startswith(prefix) and len(resolved_name) > len(prefix):
            return resolved_name[len(prefix):]

    return resolved_name


def optimizer_class(name: str):
    """Return the optimizer class for a MEALPY or project custom optimizer."""
    resolved_name = resolve_optimizer_name(name)

    if resolved_name in CUSTOM_OPTIMIZER_CLASSES:
        return CUSTOM_OPTIMIZER_CLASSES[resolved_name]

    return _available_optimizers()[resolved_name]


def is_custom_optimizer(name: str) -> bool:
    return resolve_optimizer_name(name) in CUSTOM_OPTIMIZERS


def list_available_optimizers() -> str:
    """List installed MEALPY optimizers plus project custom optimizers."""
    rows = []

    for optimizer_name in sorted(_available_optimizers(), key=str.casefold):
        rows.append((optimizer_acronym(optimizer_name), optimizer_name))

    width = max((len(display_name) for display_name, _ in rows), default=0)
    lines = [
        f"{display_name:<{width}} -> {optimizer_name}"
        for display_name, optimizer_name in rows
    ]
    lines.append("")
    lines.append("Custom:")
    lines.extend(CUSTOM_OPTIMIZERS)
    return "\n".join(lines)
