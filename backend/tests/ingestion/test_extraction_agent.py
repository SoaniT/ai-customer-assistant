# tests/ingestion/test_extraction_agent.py
"""
JSON-mode extraction agent tests. The fake model mirrors a ChatGroq bound
with response_format={"type": "json_object"}: invoke() returns an object
whose `.content` holds the model's raw JSON string.
"""

from dataclasses import dataclass

from ingestion.extraction.agent import build_extraction_agent, extract_chunk


@dataclass
class _AIMessage:
    content: str


class _FakeJsonModel:
    def __init__(self, payload: str):
        self._payload = payload
        self.bound_format = None

    def bind(self, **kwargs):
        self.bound_format = kwargs
        return self

    def invoke(self, messages):
        return _AIMessage(self._payload)


def _run(payload: str):
    agent = build_extraction_agent(_FakeJsonModel(payload))
    return extract_chunk(
        agent,
        source_name="https://example.com/",
        chunk_index=3,
        chunk_text="Alpinist provides the Alpinist Suite to Acme Co.",
    )


def test_json_mode_is_bound():
    fake = _FakeJsonModel('{"entities": [], "attributes": [], "relations": []}')
    build_extraction_agent(fake)
    assert fake.bound_format == {"response_format": {"type": "json_object"}}


def test_extract_chunk_parses_full_json():
    payload = (
        '{"entities": [{"entity_type": "Company", "name": "Acme Co", "label": "Acme Co"}],'
        ' "attributes": [{"entity_type": "Company", "entity_name": "Acme Co",'
        ' "namespace": "company", "attribute_name": "industry", "value": "Software"}],'
        ' "relations": [{"source_entity_type": "Company", "source_entity_name": "Acme Co",'
        ' "target_entity_type": "Product", "target_entity_name": "Alpinist Suite",'
        ' "relation_type": "uses"}]}'
    )
    result = _run(payload)
    assert result.chunk_index == 3
    assert result.entity == ("Company", "Acme Co")
    assert len(result.facts) == 1
    assert len(result.relations) == 1


def test_extract_chunk_empty_json_yields_empty_extraction():
    result = _run('{"entities": [], "attributes": [], "relations": []}')
    assert result.entity is None
    assert result.facts == ()
    assert result.relations == ()


def test_extract_chunk_malformed_json_yields_empty_extraction():
    result = _run("not json at all")
    assert result.entity is None
    assert result.facts == ()
    assert result.relations == ()