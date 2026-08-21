"""
ontology.py — canonicalization for the ingestion EAV extraction agent.

The static vocabulary (entity types, names, attributes, values, relations
and all synonym tables) lives in ontology_data.py. This module only holds the
matching logic that collapses any raw string the extractor emits to a single
canonical value before it reaches the database:

  * entity types: "organization" / "org" -> "Company"
  * entity names: "alpinist" / "alpinist studio" / "alpinist studios"
                  -> "Alpinist Studios"   (so all variants are ONE entity)
  * attributes:   "mail" / "e-mail"       -> "email"
  * values:       "company" / "organization" -> "Company"
  * relations:    "works for" / "hires"   -> "employs"

Persistence re-canonicalizes at write time (a backstop on every Entity /
Attribute / Value / Relation insert), so even raw strings that bypass the
extract-time normalization still merge into one canonical row.

Public API:

    canonicalize_entity_type(name) -> CanonicalizationResult     (raises on unknown)
    canonicalize_entity_name(name) -> CanonicalizationResult     (falls back to input)
    canonicalize_value(value) -> CanonicalizationResult          (falls back to input)
    canonicalize_attribute(type, name) -> CanonicalizationResult (raises on unknown)
    canonicalize_relation_type(name) -> CanonicalizationResult | None
    safe_canonicalize_*                                  (never raise)
    attributes_for_entity_type(type) -> tuple[str, ...]
    value_type_for_attribute(attr) -> ValueType
    formatted_ontology_reference() -> str                (prompt block)
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional

from ingestion.extraction.ontology_data import (
    ALL_ENTITY_TYPES,
    ATTRIBUTE_SYNONYMS,
    ATTRIBUTE_VALUE_TYPES,
    DOMAIN_ENTITY_TYPES,
    ENTITY_NAME_SYNONYMS,
    ENTITY_TYPE_ATTRIBUTES,
    ENTITY_TYPE_SYNONYMS,
    RELATION_BACKED_ATTRIBUTES,
    RELATION_TYPE_SYNONYMS,
    RELATION_TYPE_VOCABULARY,
    VALUE_SYNONYMS,
)

ValueType = Literal["string", "number", "boolean", "date", "json"]


class UnknownEntityTypeError(ValueError):
    """Raised when raw text cannot be resolved to a canonical entity type."""


class UnknownAttributeError(ValueError):
    """Raised when raw text cannot be resolved to a canonical attribute."""


@dataclass(frozen=True)
class CanonicalizationResult:
    canonical_term: str
    confidence: float
    alternate_candidates: tuple[str, ...] = ()


# ==========================================================================
# Normalization + exact-match indices (no fuzzy guessing for types/names/
# values: a fuzzy hit is more likely a false positive than a real alias).
# ==========================================================================

_FUZZY_MATCH_CUTOFF: float = 0.6
_FUZZY_MATCH_COUNT: int = 3


def _normalize(text: str) -> str:
    """Lowercase, collapse hyphens/underscores to spaces and repeated
    whitespace. Pure, total, no I/O."""
    return " ".join(text.strip().lower().replace("-", " ").replace("_", " ").split())


def _normalize_name(text: str) -> str:
    """_normalize plus dropping trailing periods ("Alpinist Studios.")."""
    return _normalize(text.strip().rstrip(".").strip())


_ENTITY_TYPE_EXACT_INDEX: Mapping[str, str] = MappingProxyType(
    {
        **{_normalize(t): t for t in ALL_ENTITY_TYPES},
        **{_normalize(s): c for s, c in ENTITY_TYPE_SYNONYMS.items()},
    }
)

_ENTITY_NAME_EXACT_INDEX: Mapping[str, str] = MappingProxyType(
    {_normalize_name(s): c for s, c in ENTITY_NAME_SYNONYMS.items()}
)

_VALUE_EXACT_INDEX: Mapping[str, str] = MappingProxyType(
    {_normalize(s): c for s, c in VALUE_SYNONYMS.items()}
)

_RELATION_TYPE_EXACT_INDEX: Mapping[str, str] = MappingProxyType(
    {
        **{_normalize(t): t for t in RELATION_TYPE_VOCABULARY},
        **{_normalize(s): c for s, c in RELATION_TYPE_SYNONYMS.items()},
    }
)


def _fuzzy_confidence(normalized_candidate: str, closest_key: str) -> float:
    return round(difflib.SequenceMatcher(None, normalized_candidate, closest_key).ratio(), 2)


# ==========================================================================
# PUBLIC API
# ==========================================================================


def attributes_for_entity_type(entity_type: str) -> tuple[str, ...]:
    """Canonical attribute set for a canonical entity type."""
    try:
        return ENTITY_TYPE_ATTRIBUTES[entity_type]
    except KeyError as exc:
        raise UnknownEntityTypeError(f"{entity_type!r} is not a canonical entity type") from exc


def value_type_for_attribute(attribute: str) -> ValueType:
    """CHECK-constraint-compliant value_type for a canonical attribute."""
    try:
        return ATTRIBUTE_VALUE_TYPES[attribute]
    except KeyError as exc:
        raise UnknownAttributeError(f"no value_type mapping for attribute {attribute!r}") from exc


def canonicalize_entity_type(candidate_text: str) -> CanonicalizationResult:
    """Exact-match + synonym lookup to a canonical ``entity.entity_type``.
    Raises UnknownEntityTypeError on a miss (the ``safe_*`` wrapper then
    falls back to the raw string)."""
    normalized = _normalize(candidate_text)
    if not normalized:
        raise UnknownEntityTypeError("empty entity type", candidate_text)
    if normalized in _ENTITY_TYPE_EXACT_INDEX:
        return CanonicalizationResult(canonical_term=_ENTITY_TYPE_EXACT_INDEX[normalized], confidence=1.0)
    raise UnknownEntityTypeError(f"no canonical entity type found for {candidate_text!r}", candidate_text)


def canonicalize_entity_name(candidate_text: str) -> CanonicalizationResult:
    """Map an entity name variant to its canonical display name (e.g.
    "alpinist studio" -> "Alpinist Studios"). Unknown names pass through
    unchanged at zero confidence, so ordinary names are never rejected."""
    normalized = _normalize_name(candidate_text)
    if not normalized:
        raise UnknownEntityTypeError("empty entity name", candidate_text)
    if normalized in _ENTITY_NAME_EXACT_INDEX:
        return CanonicalizationResult(canonical_term=_ENTITY_NAME_EXACT_INDEX[normalized], confidence=1.0)
    return CanonicalizationResult(canonical_term=" ".join(candidate_text.split()), confidence=0.0)


def canonicalize_value(value: Any) -> CanonicalizationResult:
    """Map a string value variant to its canonical value (e.g. "organization"
    -> "Company"). Non-strings pass through unchanged; unmatched strings have
    whitespace collapsed. Never raises."""
    if not isinstance(value, str):
        return CanonicalizationResult(canonical_term=value, confidence=1.0)
    normalized = _normalize(value)
    if normalized and normalized in _VALUE_EXACT_INDEX:
        return CanonicalizationResult(canonical_term=_VALUE_EXACT_INDEX[normalized], confidence=1.0)
    return CanonicalizationResult(canonical_term=" ".join(value.split()), confidence=0.0)


def canonicalize_attribute(entity_type: str, candidate_text: str) -> CanonicalizationResult:
    """Exact-match + synonym lookup to a canonical attribute valid *for the
    given entity type*. Raises UnknownEntityTypeError / UnknownAttributeError."""
    valid_attributes = attributes_for_entity_type(entity_type)
    normalized = _normalize(candidate_text)
    scoped_index: Mapping[str, str] = MappingProxyType(
        {
            **{_normalize(a): a for a in valid_attributes},
            **{
                _normalize(synonym): canonical
                for synonym, canonical in ATTRIBUTE_SYNONYMS.items()
                if canonical in valid_attributes
            },
        }
    )
    if normalized in scoped_index:
        return CanonicalizationResult(canonical_term=scoped_index[normalized], confidence=1.0)
    raise UnknownAttributeError(
        f"no canonical attribute found for {candidate_text!r} on entity type {entity_type!r}",
        candidate_text,
    )


def canonicalize_relation_type(candidate_text: Optional[str]) -> Optional[CanonicalizationResult]:
    """Vocabulary/synonym match at high confidence; fuzzy match at proportional
    confidence; otherwise the slugified candidate at low confidence (free-text
    relations are schema-valid). None only for empty input."""
    if candidate_text is None or not candidate_text.strip():
        return None
    normalized = _normalize(candidate_text)
    if normalized in _RELATION_TYPE_EXACT_INDEX:
        return CanonicalizationResult(canonical_term=_RELATION_TYPE_EXACT_INDEX[normalized], confidence=1.0)
    close_keys = difflib.get_close_matches(
        normalized, _RELATION_TYPE_EXACT_INDEX.keys(), n=_FUZZY_MATCH_COUNT, cutoff=_FUZZY_MATCH_CUTOFF
    )
    if close_keys:
        resolved = tuple(dict.fromkeys(_RELATION_TYPE_EXACT_INDEX[key] for key in close_keys))
        best, *alternates = resolved
        return CanonicalizationResult(
            canonical_term=best,
            confidence=_fuzzy_confidence(normalized, close_keys[0]),
            alternate_candidates=tuple(alternates),
        )
    return CanonicalizationResult(canonical_term=normalized.replace(" ", "_"), confidence=0.30)


# --------------------------------------------------------------------------
# Non-raising wrappers used by the extractor (tools.py) and the persistence
# backstop (persistence.py): never raise, fall back to the original text so
# genuinely unknown terms keep flowing through unchanged.
# --------------------------------------------------------------------------


def safe_canonicalize_entity_type(candidate_text: Optional[str]) -> str:
    if not candidate_text or not candidate_text.strip():
        return candidate_text or ""
    try:
        return canonicalize_entity_type(candidate_text).canonical_term
    except UnknownEntityTypeError:
        return candidate_text.strip()


def safe_canonicalize_entity_name(candidate_text: Optional[str]) -> str:
    if not candidate_text or not candidate_text.strip():
        return candidate_text or ""
    return canonicalize_entity_name(candidate_text).canonical_term


def safe_canonicalize_value(value: Any) -> Any:
    return canonicalize_value(value).canonical_term


def safe_canonicalize_attribute(entity_type: Optional[str], candidate_text: Optional[str]) -> str:
    if not entity_type or not candidate_text or not candidate_text.strip():
        return candidate_text or ""
    try:
        return canonicalize_attribute(entity_type, candidate_text).canonical_term
    except (UnknownEntityTypeError, UnknownAttributeError):
        return candidate_text.strip()


def safe_canonicalize_relation_type(candidate_text: Optional[str]) -> str:
    if not candidate_text or not candidate_text.strip():
        return candidate_text or ""
    result = canonicalize_relation_type(candidate_text)
    return result.canonical_term if result else candidate_text.strip()


# --------------------------------------------------------------------------
# Prompt reference block
# --------------------------------------------------------------------------


def formatted_ontology_reference() -> str:
    """Compact Domain -> Entity Types block for the extraction system
    prompt, assembled from the static tables so it stays in sync."""
    lines = [
        f"- {domain}: {', '.join(entity_types)}"
        for domain, entity_types in DOMAIN_ENTITY_TYPES.items()
    ]
    return "\n".join(lines)


# ==========================================================================
# Module-load-time totality check: every attribute referenced by any entity
# type must have a value_type mapping, or the CHECK constraint would be
# silently violated on the first structured write.
# ==========================================================================

_ALL_REFERENCED_ATTRIBUTES: frozenset[str] = frozenset(
    attribute for attributes in ENTITY_TYPE_ATTRIBUTES.values() for attribute in attributes
)
_MISSING_VALUE_TYPES: frozenset[str] = _ALL_REFERENCED_ATTRIBUTES - frozenset(ATTRIBUTE_VALUE_TYPES.keys())
if _MISSING_VALUE_TYPES:
    raise AssertionError(
        f"ingestion ontology is inconsistent: attributes {sorted(_MISSING_VALUE_TYPES)} are used in "
        "ENTITY_TYPE_ATTRIBUTES but have no entry in ATTRIBUTE_VALUE_TYPES"
    )