"""Regression tests for the ingestion ontology's canonicalization of the
entity-type / entity-name / value / attribute / relation synonym tables.

These lock in the P0 data-quality fixes: synonyms collapse to ONE canonical
spelling so the write path (and the reconciliation migration) can merge
previously-duplicated rows.
"""

from ingestion.extraction.ontology import (
    RELATION_BACKED_ATTRIBUTES,
    safe_canonicalize_attribute,
    safe_canonicalize_entity_name,
    safe_canonicalize_entity_type,
    safe_canonicalize_relation_type,
    safe_canonicalize_value,
)


def test_entity_type_synonyms_collapse_to_canonical():
    assert safe_canonicalize_entity_type("tech_stack") == "Technology"
    assert safe_canonicalize_entity_type("tech stack") == "Technology"
    assert safe_canonicalize_entity_type("organization") == "Company"
    assert safe_canonicalize_entity_type("consulting") == "Consulting Service"


def test_entity_name_synonyms_collapse_to_single_display_name():
    for variant in ("alpinist", "alpinist studio", "alpinist studios", "Alpinist Studios."):
        assert safe_canonicalize_entity_name(variant) == "Alpinist Studios"
    assert safe_canonicalize_entity_name("laravel framework") == "Laravel"
    assert safe_canonicalize_entity_name("nodejs") == "Node.js"


def test_value_synonyms_collapse_case_and_wording_variants():
    assert safe_canonicalize_value("consulting") == "Consulting"
    assert safe_canonicalize_value("mvp development") == "MVP Development"
    assert safe_canonicalize_value("staff augmentation") == "Staff Augmentation"
    assert safe_canonicalize_value("In Progress") == "In Progress"
    assert safe_canonicalize_value("unknown") == "Unknown"


def test_relation_type_synonyms_fold_free_text_onto_one_spelling():
    assert safe_canonicalize_relation_type("initiated") == "initiates"
    assert safe_canonicalize_relation_type("developed by") == "develops"
    assert safe_canonicalize_relation_type("formerly employed at") == "formerly_employed_at"
    assert safe_canonicalize_relation_type("builds") == "develops"
    assert safe_canonicalize_relation_type("works for") == "employs"


def test_scoped_attribute_synonyms():
    assert safe_canonicalize_attribute("Person", "mail") == "email"
    assert safe_canonicalize_attribute("Person", "job title") == "title"
    assert safe_canonicalize_attribute("Company", "industry") == "industry"


def test_relation_backed_attributes_cover_double_storage_names():
    for attr in ("service", "services", "technology", "tech_stack", "clients", "products"):
        assert attr in RELATION_BACKED_ATTRIBUTES