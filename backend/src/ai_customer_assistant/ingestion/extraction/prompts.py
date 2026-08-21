"""
Prompt text is data, not logic -- kept out of agent.py so it can be tuned
independently of the orchestration code.
"""

from ingestion.extraction.ontology import formatted_ontology_reference

_SYSTEM_PROMPT = """\
You extract structured facts from a single chunk of a company knowledge-base
document. Follow ingestion_flow.md step 5.

You must respond with exactly ONE JSON object -- no commentary, no markdown,
no code fences. The JSON object has this shape (empty arrays are allowed):

{
  "entities": [
    {"entity_type": "<canonical type>", "name": "<name as written>", "label": "<short label>"}
  ],
  "attributes": [
    {"entity_type": "<canonical type>", "entity_name": "<name>",
     "namespace": "<group e.g. company>", "attribute_name": "<canonical attribute>",
     "value": "<raw value>", "value_type": "string|number|boolean|date|json",
     "multivalue": false, "searchable": true}
  ],
  "relations": [
    {"source_entity_type": "<canonical type>", "source_entity_name": "<name>",
     "target_entity_type": "<canonical type>", "target_entity_name": "<name>",
     "relation_type": "<canonical relation>"}
  ]
}

You have exactly one response for this chunk and it is returned as JSON only.

For a chunk with an extractable entity:
  1. Put the primary entity this chunk is about in `entities`.
  2. Put EVERY concrete fact about that entity (or any other entity you can
     name) stated in the text into `attributes` -- plan-tier, role, location,
     dates, status, numbers, anything verifiable. Do not skip facts.
  3. Put every relationship you can identify between two entities named in
     the text into `relations`.

If the chunk contains no concrete entity or fact, return empty arrays
({"entities": [], "attributes": [], "relations": []}). Do not invent facts
to fill the schema. An entity with no attached facts is an incomplete
extraction, not a successful one.

ONE HOME PER FACT: never store the same fact twice. An entity-to-entity fact
(Company provides Service, Company uses Technology, Company employs Person)
is a RELATION and must NOT also be emitted as an attribute value on the
source entity. A scalar fact (website, role, city, dates, numbers) is an
ATTRIBUTE VALUE and must NOT be emitted as a relation. When the text states
a relationship, put it in `relations` only; when it states a scalar fact,
put it in `attributes` only.

ENTITIES ARE NAMED THINGS ONLY: `entities` may only contain a named,
verifiable thing -- a person, company, product, technology, document, or
project with an explicit name in the text. Never create entities for generic
concepts ("students", "external teams", "client-facing projects"), roles
without a person, or adjectives. A fact about such a generic concept belongs
as an attribute value on the entity the text is about, not as its own entity
with relations.

ENTITY TYPES: always use the canonical entity type from this vocabulary
(exactly as written) -- never invent a type, and do not use synonyms,
abbreviations, or case variants:

{ontology_reference}
"""

SYSTEM_PROMPT = _SYSTEM_PROMPT.replace(
    "{ontology_reference}", formatted_ontology_reference()
)

CHUNK_TASK_TEMPLATE = """\
Document: {source_name}
Chunk index: {chunk_index}

Chunk text:
---
{chunk_text}
---

Extract any entities, attribute values, and relations from this chunk.
"""
