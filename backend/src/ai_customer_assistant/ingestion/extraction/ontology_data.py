"""
ontology_data.py — static vocabulary for the ingestion EAV extraction agent.

Everything here is plain data (no logic): the canonical entity types, their
attribute sets, value types, relation vocabulary, and the synonym tables that
collapse informal wording down to a single canonical value. The matching
logic lives in ontology.py, which imports these tables.

Synonym maps are keyed by the *raw* wording as the LLM might emit it; they
are normalized (lowercased / whitespace-collapsed) at index-build time in
ontology.py. Add a new variant by appending to the relevant map — no code
changes needed.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

ValueType = str  # "string" | "number" | "boolean" | "date" | "json"

# --------------------------------------------------------------------------
# DOMAIN -> ENTITY TYPE (application-level grouping only, never persisted)
# --------------------------------------------------------------------------

DOMAIN_ENTITY_TYPES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "Organization": (
            "Company", "Department", "Team", "Office", "Employee", "Person",
            "Role", "Client", "Partner", "Vendor",
        ),
        "Business": (
            "Service", "Consulting Service", "Support Plan", "Pricing Plan",
            "SLA", "Package", "Proposal", "Product", "Project", "Feature",
            "Module", "Component", "API", "Microservice", "Integration",
        ),
        "Technology": (
            "Technology", "Programming Language", "Framework", "Library",
            "Database", "Cloud Platform", "DevOps Tool", "Operating System",
            "Messaging System", "Vector Database",
        ),
        "Software Engineering": (
            "Architecture Pattern", "Design Pattern", "SDLC Phase",
            "Development Process", "Testing Strategy", "Deployment Strategy",
            "Coding Standard", "Git Workflow", "CI/CD Pipeline",
        ),
        "Knowledge": (
            "Knowledge Source", "Knowledge Category", "Document", "FAQ",
            "Glossary Term", "Knowledge Chunk",
        ),
        "Policies": (
            "Policy", "Guideline", "Standard", "Procedure",
            "Employee Handbook", "Training Material", "Benefit", "Leave Type",
        ),
        "Security": (
            "Security Policy", "Security Practice", "Compliance Standard",
            "Authentication Method", "Authorization Method", "Incident",
            "Risk", "Vulnerability",
        ),
        "Infrastructure": (
            "Server", "Environment", "Container", "Cluster", "Storage",
            "Network", "Monitoring Tool", "Backup Strategy",
        ),
        "Business Intelligence": (
            "Industry", "Business Process", "Workflow", "Customer Type",
            "Business Goal", "KPI", "Metric",
        ),
        "Artificial Intelligence": (
            "AI Solution", "RAG Pipeline", "Embedding Model", "AI Model",
            "LLM", "Prompt Template", "Knowledge Graph", "Agent",
            "Workflow Agent",
        ),
        "Documentation": (
            "Case Study", "Whitepaper", "Report", "Meeting", "Release Note",
            "Changelog",
        ),
    }
)

ALL_ENTITY_TYPES: tuple[str, ...] = tuple(
    entity_type
    for entity_types in DOMAIN_ENTITY_TYPES.values()
    for entity_type in entity_types
)

# --------------------------------------------------------------------------
# ENTITY TYPE -> ATTRIBUTES (attribute.namespace = entity type)
# --------------------------------------------------------------------------

_BASE_ENTITY_ATTRIBUTES: tuple[str, ...] = (
    "name", "description", "status", "tags", "created_at", "updated_at",
)

_ENTITY_TYPE_ATTRIBUTE_EXTENSIONS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "Company": ("industry", "website", "country", "city"),
        "Department": ("owner", "team"),
        "Team": ("manager", "department"),
        "Office": ("country", "city", "address"),
        "Employee": ("role", "department", "email", "phone"),
        "Person": ("role", "title", "employer", "department", "email", "phone"),
        "Role": ("department",),
        "Client": ("industry", "contract_type", "website"),
        "Partner": ("industry", "contract_type", "website"),
        "Vendor": ("industry", "contract_type", "website"),
        "Service": ("service_level", "pricing_model", "support_level"),
        "Consulting Service": ("service_level", "pricing_model", "estimated_duration"),
        "Support Plan": ("support_level", "pricing_model", "response_time"),
        "Pricing Plan": ("pricing_model", "cost", "budget"),
        "SLA": ("service_level", "response_time", "uptime"),
        "Package": ("pricing_model", "version"),
        "Proposal": ("target_customer", "estimated_duration", "cost"),
        "Product": ("category", "version", "pricing_model"),
        "Project": ("owner", "start_date", "end_date", "budget"),
        "Feature": ("version", "dependencies"),
        "Module": ("version", "dependencies", "programming_language"),
        "Component": ("version", "dependencies", "technology"),
        "API": ("api_type", "protocol", "authentication_method", "version"),
        "Microservice": ("technology", "programming_language", "protocol", "deployment_model"),
        "Integration": ("provider", "authentication_method", "protocol"),
        "Technology": ("category", "version"),
        "Programming Language": ("version",),
        "Framework": ("programming_language", "version"),
        "Library": ("programming_language", "version"),
        "Database": ("technology", "version", "deployment_model"),
        "Cloud Platform": ("provider", "region"),
        "DevOps Tool": ("category", "version"),
        "Operating System": ("version",),
        "Messaging System": ("protocol", "technology"),
        "Vector Database": ("embedding_dimension", "technology", "deployment_model"),
        "Architecture Pattern": ("category",),
        "Design Pattern": ("category",),
        "SDLC Phase": ("category",),
        "Development Process": ("category",),
        "Testing Strategy": ("category",),
        "Deployment Strategy": ("environment", "category"),
        "Coding Standard": ("programming_language", "category"),
        "Git Workflow": ("category",),
        "CI/CD Pipeline": ("environment", "technology"),
        "Knowledge Source": ("source_type", "file_type", "checksum"),
        "Knowledge Category": (),
        "Document": ("document_type", "file_type", "source"),
        "FAQ": ("category",),
        "Glossary Term": ("category",),
        "Knowledge Chunk": ("source", "version_number"),
        "Policy": ("category", "effective_date", "review_date"),
        "Guideline": ("category",),
        "Standard": ("category", "compliance"),
        "Procedure": ("category", "department"),
        "Employee Handbook": ("effective_date", "version"),
        "Training Material": ("category", "document_type"),
        "Benefit": ("category",),
        "Leave Type": ("category",),
        "Security Policy": ("compliance", "effective_date", "review_date"),
        "Security Practice": ("category", "compliance"),
        "Compliance Standard": ("compliance", "review_date"),
        "Authentication Method": ("category", "protocol"),
        "Authorization Method": ("category", "protocol"),
        "Incident": ("severity",),
        "Risk": ("risk_level", "category"),
        "Vulnerability": ("severity",),
        "Server": ("environment", "region", "operating_system"),
        "Environment": ("region", "category"),
        "Container": ("technology", "environment"),
        "Cluster": ("environment", "region", "scalability"),
        "Storage": ("storage_type", "region"),
        "Network": ("protocol", "region"),
        "Monitoring Tool": ("category", "technology"),
        "Backup Strategy": ("backup_frequency", "storage_type"),
        "Industry": ("category",),
        "Business Process": ("category", "owner"),
        "Workflow": ("category", "owner"),
        "Customer Type": ("category",),
        "Business Goal": ("category", "owner"),
        "KPI": ("category", "performance"),
        "Metric": ("category", "performance"),
        "AI Solution": ("category", "provider", "llm_provider"),
        "RAG Pipeline": ("retrieval_method", "embedding_dimension", "vector_database"),
        "Embedding Model": ("provider", "embedding_dimension"),
        "AI Model": ("provider", "model_name", "version"),
        "LLM": ("provider", "model_name", "version"),
        "Prompt Template": ("category", "version"),
        "Knowledge Graph": ("category", "technology"),
        "Agent": ("category", "llm_provider"),
        "Workflow Agent": ("category", "llm_provider"),
        "Case Study": ("category", "industry"),
        "Whitepaper": ("category", "release_date"),
        "Report": ("category", "release_date"),
        "Meeting": ("category", "start_date"),
        "Release Note": ("version", "release_date"),
        "Changelog": ("version", "release_date"),
    }
)

ENTITY_TYPE_ATTRIBUTES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        entity_type: _BASE_ENTITY_ATTRIBUTES + extra
        for entity_type, extra in _ENTITY_TYPE_ATTRIBUTE_EXTENSIONS.items()
    }
)

# --------------------------------------------------------------------------
# ATTRIBUTE -> VALUE_TYPE (must satisfy attribute.value_type CHECK
# constraint: string | number | boolean | date | json)
# --------------------------------------------------------------------------

ATTRIBUTE_VALUE_TYPES: Mapping[str, ValueType] = MappingProxyType(
    {
        "name": "string", "title": "string", "description": "string",
        "summary": "string", "category": "string", "subtype": "string",
        "status": "string", "priority": "string", "version": "string",
        "tags": "json", "role": "string", "employer": "string",
        "owner": "string", "department": "string", "team": "string",
        "manager": "string", "assigned_to": "string", "maintained_by": "string",
        "created_by": "string", "approved_by": "string",
        "created_at": "date", "updated_at": "date", "effective_date": "date",
        "review_date": "date", "expiry_date": "date", "start_date": "date",
        "end_date": "date", "release_date": "date",
        "industry": "string", "business_model": "string",
        "target_customer": "string", "pricing_model": "string",
        "support_level": "string", "service_level": "string",
        "estimated_duration": "string", "contract_type": "string",
        "technology": "string", "programming_language": "string",
        "framework": "string", "database": "string", "cloud_provider": "string",
        "deployment_model": "string", "architecture": "string",
        "protocol": "string", "api_type": "string",
        "authentication_method": "string",
        "encryption": "string", "compliance": "string", "severity": "string",
        "confidentiality": "string", "availability": "string",
        "integrity": "string", "risk_level": "string",
        "document_type": "string", "source": "string", "source_type": "string",
        "file_type": "string", "checksum": "string", "version_number": "number",
        "model_name": "string", "provider": "string",
        "embedding_dimension": "number", "vector_database": "string",
        "chunk_size": "number", "retrieval_method": "string",
        "llm_provider": "string",
        "environment": "string", "region": "string",
        "operating_system": "string", "storage_type": "string",
        "backup_frequency": "string", "monitoring_tool": "string",
        "performance": "string", "response_time": "number", "uptime": "number",
        "scalability": "string", "cost": "number", "budget": "number",
        "website": "string", "email": "string", "phone": "string",
        "address": "string", "country": "string", "city": "string",
        "notes": "string", "remarks": "string", "dependencies": "json",
        "prerequisites": "json", "related_document": "string",
        "reference": "string", "active": "boolean",
    }
)

# --------------------------------------------------------------------------
# RELATIONS — advisory vocabulary (relation.relation_type has no CHECK
# constraint, so this is normalization, not a hard gate).
# --------------------------------------------------------------------------

RELATION_TYPE_VOCABULARY: tuple[str, ...] = (
    "uses", "depends_on", "implements", "belongs_to", "managed_by",
    "owned_by", "created_by", "approved_by", "integrates_with", "contains",
    "requires", "supports", "deployed_on", "stored_in", "hosted_on",
    "communicates_with", "related_to", "employs", "provides", "develops",
    "mentions", "initiates", "formerly_employed_at",
)

# Attribute names whose data represents entity-to-entity facts (a "provides
# X"/"uses X" relation), so the same fact must NOT also be stored as a scalar
# value. persistence.py uses this to drop the redundant representation.
RELATION_BACKED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "service", "services", "technology", "tech_stack", "tools", "stack",
        "languages", "frameworks", "integrations", "clients", "products",
        "projects", "competitors", "partners", "vendors",
    }
)

# --------------------------------------------------------------------------
# SYNONYMS — informal wording -> one canonical value. Extend any of these
# maps to collapse more variants; nothing else in the code needs to change.
# --------------------------------------------------------------------------

# Entity types: the LLM may say "organization" where the canonical type is
# "Company". Collapsing these is what stops the same real-world entity from
# being stored once per type label.
ENTITY_TYPE_SYNONYMS: Mapping[str, str] = MappingProxyType(
    {
        "db": "Database",
        "backend framework": "Framework",
        "vector db": "Vector Database",
        "ai model": "AI Model",
        "chat model": "LLM",
        "embedding": "Embedding Model",
        "api endpoint": "API",
        "micro-service": "Microservice",
        "microservice": "Microservice",
        "release docs": "Release Note",
        "user manual": "Training Material",
        "handbook": "Employee Handbook",
        "faq": "FAQ",
        "kb article": "Document",
        "knowledge base article": "Document",
        "sla": "SLA",
        "org": "Company",
        "company": "Company",
        "organization": "Company",
        "organisation": "Company",
        "corporation": "Company",
        "firm": "Company",
        "startup": "Company",
        "person": "Person",
        "people": "Person",
        "individual": "Person",
        "employee": "Employee",
        "staff": "Employee",
        "team member": "Employee",
        "sector": "Industry",
        "industry": "Industry",
        "market segment": "Industry",
        "non-profit": "Industry",
        "nonprofits": "Industry",
        "non profits": "Industry",
        "technology": "Technology",
        "service": "Service",
        "consulting": "Consulting Service",
        "consulting service": "Consulting Service",
        "product": "Product",
        "project": "Project",
        "tech stack": "Technology",
        "tech_stack": "Technology",
    }
)

# Entity names: the LLM may call the same real-world entity by different
# names. Every variant here collapses to the one canonical display name, so
# "alpinist studio", "alpinist studios" and "alpinist" all resolve to the
# same entity row. Canonical names are title-cased.
ENTITY_NAME_SYNONYMS: Mapping[str, str] = MappingProxyType(
    {
        "alpinist": "Alpinist Studios",
        "alpinist studio": "Alpinist Studios",
        "alpinist studios": "Alpinist Studios",
        "alpinist studio inc": "Alpinist Studios",
        "alpinist studios inc": "Alpinist Studios",
        "alpinist studios pvt ltd": "Alpinist Studios",
        "laravel framework": "Laravel",
        "react js": "React",
        "nodejs": "Node.js",
        "postgres": "PostgreSQL",
        "mongodb": "MongoDB",
        "company": "Company",
        "organization": "Company",
        "organisation": "Company",
        "corporation": "Company",
        "firm": "Company",
    }
)

# Attribute values: synonyms for common fact values (e.g. status or
# organization wording), so "company" and "organization" written as a *value*
# also collapse to one row instead of two Value rows.
VALUE_SYNONYMS: Mapping[str, str] = MappingProxyType(
    {
        "company": "Company",
        "organization": "Company",
        "organisation": "Company",
        "corporation": "Company",
        "firm": "Company",
        "n/a": "N/A",
        "not available": "N/A",
        "unknown": "Unknown",
        "in progress": "In Progress",
        "in-progress": "In Progress",
        "ongoing": "In Progress",
        "not started": "Not Started",
        "consulting": "Consulting",
        "mvp development": "MVP Development",
        "mvp dev": "MVP Development",
        "staff augmentation": "Staff Augmentation",
        "staffing": "Staff Augmentation",
        "startup facilitation": "Startup Facilitation",
        "startup support": "Startup Facilitation",
        "external teams": "External Teams",
        "external team": "External Teams",
        "in-house teams": "In-house Teams",
        "backend": "Backend",
        "frontend": "Frontend",
        "full stack": "Full Stack",
        "fullstack": "Full Stack",
        "devops": "DevOps",
        "data science": "Data Science",
        "data engineering": "Data Engineering",
        "product design": "Product Design",
        "ui/ux design": "UI/UX Design",
        "qa testing": "QA Testing",
        "quality assurance": "QA Testing",
    }
)

# Attributes: informal wording -> canonical attribute name.
ATTRIBUTE_SYNONYMS: Mapping[str, str] = MappingProxyType(
    {
        "phone number": "phone",
        "mobile": "phone",
        "mobile number": "phone",
        "mail": "email",
        "e-mail": "email",
        "created on": "created_at",
        "date created": "created_at",
        "last modified": "updated_at",
        "modified on": "updated_at",
        "owner name": "owner",
        "assigned to": "assigned_to",
        "point of contact": "owner",
        "job title": "title",
        "company": "employer",
        "organisation": "employer",
        "organization": "employer",
        "works at": "employer",
        "field": "category",
        "segment": "category",
        "services": "service",
    }
)

RELATION_TYPE_SYNONYMS: Mapping[str, str] = MappingProxyType(
    {
        "depends on": "depends_on",
        "belongs to": "belongs_to",
        "part of": "belongs_to",
        "managed by": "managed_by",
        "owned by": "owned_by",
        "created by": "created_by",
        "approved by": "approved_by",
        "integrates with": "integrates_with",
        "requires": "requires",
        "runs on": "deployed_on",
        "deployed on": "deployed_on",
        "stored in": "stored_in",
        "hosted on": "hosted_on",
        "talks to": "communicates_with",
        "connects to": "communicates_with",
        "communicates with": "communicates_with",
        "related to": "related_to",
        "employs": "employs",
        "works for": "employs",
        "works at": "employs",
        "is employed by": "employs",
        "hires": "employs",
        "provides": "provides",
        "offers": "provides",
        "sells": "provides",
        "targets": "targets",
        "serves": "serves",
        "leads": "leads",
        "headed by": "leads",
        "founded by": "founded_by",
        "founder of": "founded_by",
        "develops": "develops",
        "develop": "develops",
        "developed": "develops",
        "developed by": "develops",
        "builds": "develops",
        "mentions": "mentions",
        "mention": "mentions",
        "mentioned": "mentions",
        "mentions of": "mentions",
        "initiates": "initiates",
        "initiate": "initiates",
        "initiated": "initiates",
        "started by": "initiates",
        "formerly_employed_at": "formerly_employed_at",
        "formerly employed at": "formerly_employed_at",
        "former employer": "formerly_employed_at",
        "previously worked for": "formerly_employed_at",
        "previously worked at": "formerly_employed_at",
    }
)