.PHONY: run test

run:
	uv run --directory backend python -m ai-customer-assistant.main

test:
	uv run --project backend pytest backend/tests

crawler:
	uv run --directory backend python -m ai-customer-assistant.ingestion.crawler https://alpiniststudios.com/