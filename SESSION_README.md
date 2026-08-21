# SESSION_README — AI Customer Assistant

Long-running working memory for the AI Customer Assistant project. Covers the
entire build (since Jul 27), the recent ingestion/chat debugging session, and
current state. Update after every work block.

---

## 1. Project Overview

- **Goal:** an ingestion + chat assistant stack: crawl/upload docs -> chunk ->
  embed -> EAV knowledge graph (entities/attributes/relations) -> chat via a
  LangGraph Supervisor that routes to a Knowledge agent (hybrid structured +
  vector RAG), a Safety gate (groundedness), and a Ticket agent.
- **Stack:** Python 3.12 / FastAPI / LangGraph / SQLAlchemy async / Postgres +
  pgvector + MinIO + Tika; React/Vite frontend; Docker Compose.
- **Repo:** `/Users/nikeshbista/My Files/AI Intership - Alpinist/Week 10-12/Project/ai-customer-assistant`
- **Branch:** `feature/updated_ontology` (HEAD `e380c79`). Large pile of work
  is UNCOMMITTED — commit early and often (work was lost once already).

## 2. Timeline

### 2.1 Git history (Jul 27 -> Aug 17)

| Date | Commit/PR | What |
|---|---|---|
| Jul 27 | (repos/setup) | Project scaffold, docker-compose, backend/frontend skeleton |
| ~Jul 30 | | Ingestion pipeline: upload + URL crawl + MinIO + Tika |
| ~Aug 2 | | Chunking + embedding (BAAI/bge-base-en-v1.5), pgvector |
| ~Aug 5 | | EAV extraction (entities/attributes/relations), ontology + synonyms |
| ~Aug 7 | | Agent structure PRs: Knowledge RAG agent, Safety agent, Ticket agent |
| Aug 13 | `7f25425` "prompt implemented" | Prompt templates (rewrite/extraction/answer) |
| Aug 14 | PR #22 `fa0c378` | agent-structure merged |
| Aug 17 | PR #24 `e380c79` | frontend merged |

### 2.2 Earlier assistant sessions (pre-chat build)

- Designed and built the full agents_integration_plan stack: Supervisor graph
  (classifier -> routing -> Knowledge/Safety/Ticket), checkpointer-backed
  threads, `/chat` endpoint, BGE shared singleton, safety gate.
- Wrote the Knowledge agent: rewrite -> extraction -> strategy edge
  (structured/vector/hybrid) -> structured_lookup + vector_search -> rank ->
  deduplicate -> context -> prompt -> answer LLM -> GroundedResponse.
- Built EAV extraction + ontology data + synonym normalization + persistence.

### 2.3 Pre-chat debugging + model saga (this session, Aug 17-18)

- **P0:** EAV extraction for the about page returned nothing (entity_id nulls /
  empty graph) — fixed via new ontology data + extraction flow rework.
- **Double execution:** crawler pipeline was running extraction twice (agent +
  pipeline) — deduped to run once.
- **Model saga:** `llama-3.3-70b-versatile` / `llama-3.1-8b-instant` are
  unavailable on all 3 GROQ keys. Backend falls back to `EAV_MODEL=openai/gpt-oss-120b`
  (`.env`). Container env is baked at build; `.env` changes need a rebuild.
- Chunk unique constraint moved out of `db/models.py` into DB migration
  `c7d5a3b2e1f0` (DO NOT touch `db/models.py` for this).

## 3. Architecture (current)

- **Ingestion:** `ingestion/` — `pipeline.py` orchestrates fetch (Playwright/
  trafilatura) -> markdown -> chunk (`chunk_embed/`) -> embed -> extract EAV
  (single JSON completion; extraction is NOT an agent) -> persist
  (`persistence.py`, `queue/`).
- **Crawler enrichment** (`ingestion/crawler/extractor.py`): prepends
  `# <site name>` (from og:site_name/title/domain) and appends
  `## Team Members` (member-name/member-designation cards) so navbar/team
  content survives trafilatura. Re-crawling an unchanged page now creates a
  NEW version (checksums changed) instead of `duplicate_skipped`.
- **Knowledge agent** (`agents/knowledge/`): rewrite -> extraction ->
  strategy -> retrieval -> answer. `config.py` is `KnowledgeAgentConfig`
  (frozen, env prefix `KNOWLEDGE_AGENT_`).
- **Supervisor** (`agents/supervisor/`): classifier (Groq, `prompt.py` with
  `DEFAULT_DOMAIN_DEFINITION` + `build_supervisor_system_prompt()`), routing
  (`routing.py`, `_DECLINE_RESPONSE` at line 40), nodes.
- **Safety gate** (`agents/safety_agent/`): per-sentence BGE groundedness
  (sentence_threshold 0.75, aggregation 0.80); fallback template in
  `fallback_response.py`.
- **Chat:** `services/chat_service.py` builds everything once; `/chat`
  requires `thread_id`. Escalation/escalation-confirm uses `interrupt()`.

## 4. Known issues / gotchas

- **Retrieval:** `vector_search` raises `EmptyRetrievalError` when no chunk
  clears `similarity_threshold=0.7`. NEW: a relaxed second pass at
  `relaxed_similarity_threshold=0.55` runs on a primary miss, recovering
  count-style queries (e.g. "how many members" scores 0.666 against the team
  chunk). Both thresholds are env-tunable (`KNOWLEDGE_AGENT_*`).
- **Escalation interrupt leak:** after an ungrounded answer the graph pauses
  for confirmation; a NEW question sent to the same thread is consumed as the
  resume value (not a new message). Test knowledge questions in fresh threads.
- **Rate limit (operational constraint):** `openai/gpt-oss-120b` on the
  on-demand tier is capped at **200k tokens/day** for the Groq org. A long
  chat/testing session exhausts it (each message ~3-6k tokens across
  classifier + rewrite + extraction + answer), after which every Groq call
  429s. This caused the "HTTP 500" the user saw: the classifier raised
  `groq.RateLimitError` and nothing handled it. FIXED: (a) both Groq clients
  retry short-wait 429s (`Retry-After` <= 5s) and re-raise long-wait ones;
  (b) `classify_and_route` degrades a classifier failure to the safe-fallback
  decision instead of crashing; (c) the `/chat` route catches any residual
  exception and returns the safe fallback — no more HTTP 500s. Verified live
  under a real 429 (HTTP 200 + fallback text). Consider upgrading the Groq
  tier or rotating to another key when the daily budget resets.
- **Ungrounded on answerable questions (partially fixed):** "is Katherine Mah
  an advisor?" returned the ungrounded fallback because the vector pass missed
  the declarative chunk text AND extraction didn't resolve the person entity —
  so the strategy went vector-only → empty retrieval. The EAV graph HAS the
  answer (`Katherine Mah` -> `holds_role` -> `Advisor`). FIXED with a
  deterministic `entity_name_fallback` in `structured_lookup.py` + wired into
  the vector-search node: when vector retrieval is empty, the longest known
  entity name in the query resolves to that entity's attributes + all outgoing
  relations. Verified live against Postgres. "who is the founder?" (context
  had the answer but the answer LLM self-judged ungrounded) remains
  LLM-stochastic; the real groundedness check isn't wired in yet.
- **"what is alpinist?" classified out-of-scope (fix pending model
  verification):** borderline classifier call. Added company-overview /
  "what is <company>" wording to `DEFAULT_DOMAIN_DEFINITION` in
  `supervisor/prompt.py`. Needs the rate limit to reset to verify.
- **Checkpointer warning** (informational): "Deserializing unregistered type
  agents.contracts.ConversationTurn... Set LANGGRAPH_STRICT_MSGPACK=true...".
  Not addressed.
- **gpt-oss-120b weakness:** borderline-generic entities (roles as entities,
  "Consulting" as a service), occasional nondeterministic `is_grounded=false`
  on grounded answers (answer LLM self-judges; live safety gate trusts the
  knowledge `is_grounded` hint since no real check is injected in `main.py`).
- **Groq `output_parse_failed` 400 (FIXED):** gpt-oss-120b under
  `response_format={"type": "json_object"}` (or strict `json_schema`) emits
  output Groq can't parse once conversation history grows past ~3-4 turns —
  crashed the classifier/knowledge calls after 2-3 chats. Fix: drop
  `response_format` entirely (free-form output parses 100% reliably), cap the
  classifier history at the last 6 turns, strip code fences in
  `parse_llm_response`, and bounded-retry `output_parse_failed` +
  `APIConnectionError`/`APITimeoutError` in both `GroqSupervisorLLMClient._complete`
  and `GroqKnowledgeProvider._complete`.
- **Request timed out (FIXED):** the frontend aborted `/chat` after
  `DEFAULT_TIMEOUT=15s` (`frontend/src/api.js`), but the backend knowledge
  adapter allowed 30s and later-turn answers grew to 20-40s (gpt-oss-120b is
  slow + verbose; history grows prompts). Fixes: chat.js passes
  `{ timeout: 90000 }`; knowledge adapter timeout now 60s default /
  `KNOWLEDGE_AGENT_TIMEOUT_S` env (`agents_wiring.py`); knowledge-graph
  conversation history capped at last 6 turns (`_KNOWLEDGE_MAX_HISTORY_TURNS`);
  `max_tokens=1024` bounds generation on both Groq clients. Verified 4-5-turn
  threads complete in ~3-31s with no timeout.
- **Escalation interrupt leak (FIXED):** after an ungrounded answer the graph
  paused for confirmation; a NEW question sent to the same thread was consumed
  as the resume value (not a new message), echoing the stale fallback. Fix in
  `chat_service.handle_message_turn`: when `snapshot.next` is the
  `safety_gate` node (escalation pending) and the message is NOT an
  affirmative, decline first (`Command(resume=False)`) then run the message as
  a fresh turn. Affirmatives still resume to the ticket/email flow. Verified
  end-to-end both ways; 2 new unit tests.
- **Pre-existing test failures** (not ours): Playwright browser not installed
  (`test_crawler.py`, `test_fetcher.py`) and
  `test_chunking.py::TestStructuredSourceType::test_one_chunk_no_splitting_regardless_of_size`.
- Container code is baked at build; code changes need
  `docker compose up -d --build --no-deps backend worker` (postgres/minio/
  tika untouched). Backend startup ~70s (BGE download).
- Tests run from `backend/` with `.venv/bin/python -m pytest` (tests aren't
  mounted into the container).

## 5. Test suite state

- Latest: **334 passed, 2 skipped, 2 failed + 4 errors** (the failed/error
  tests are the pre-existing Playwright/chunking issues above).
- New tests this session: `tests/ingestion/test_extractor.py` (8),
  `tests/agents/knowledge/test_vector_search.py` (5, relaxed-pass),
  `tests/agents/knowledge/test_entity_name_fallback.py` (15, name fallback,
  multi-entity subject-first, source-inventory, member-roster),
  `tests/agents/knowledge/test_nodes_fallback.py` (5, node-level fallback
  wiring + roster-gate regression), `tests/agents/knowledge/test_structured_facts_reducer.py`
  (1, parallel-write reducer regression),
  `tests/agents/supervisor_agent_test/test_groq_llm_client.py` (6, history
  cap + parse-failure retry + rate-limit handling), +1 history-cap test in
  `test_agents_wiring.py`, +2 escalation-interrupt tests in
  `test_chat_service.py`, +1 classifier-degradation test in
  `test_supervisor.py`, +2 supervisor prompt tests and +1 code-fence parser
  test in `test_supervisor.py`.

## 6. Current DB state (Postgres, `feature/updated_ontology`)

- Sources: 8 active `knowledge_source` rows (homepage, /about, /services,
  /career, /contact, /soani-tech-is-now-alpinist-studios,
  /from-student-to-studio, +1 more); source-inventory answers now report
  "8 links".
- KB graph: Alpinist Studios (Company, ~73 attribute facts) + 19 Person + 9
  Role + 8 Service + 11 Technology + 4 Project + 3 Company + 1 Framework;
  ~31 relations (`holds_role`, `provides`, `employs`, ...). No pricing data
  anywhere in the KB (only "Cost-efficient" branding) — the honest ungrounded
  answer for "how much would it cost" is correct.
- Chat verified: CEO / services / members / "is <person> <role>" all answer
  with citations; weather OUT_OF_SCOPE.

## 7. Recent work this session (Aug 18)

1. SESSION_README created; Q&A (4 agents; extraction not an agent).
2. Root-caused "cannot crawl /about" -> trafilatura strips navbar/team cards +
   checksum dedup -> enriched extractor (site name + team members).
3. Root-caused chat decline -> `{DOMAIN_DEFINITION}` placeholder never filled
   in Supervisor prompt -> rewrote `prompt.py` (builder + env override).
4. Root-caused "how many members" fallback -> 0.666 < 0.7 retrieval threshold
   -> added relaxed second pass in `vector_search` (config-gated, tested).
5. Rebuilt containers; verified all three queries end-to-end via `/chat`.
6. Fixed Groq `output_parse_failed` 400 after 2-3 chats: root cause was
   `response_format=json_object` + growing conversation history on
   gpt-oss-120b. Fix: free-form calls + history cap (6 turns) + code-fence
   stripping + bounded retry (parse-fail + connection/timeout) in the
   Supervisor classifier and Knowledge provider. Verified 13/13 classifier
   turns and a full 5-message chat thread with no 400s.
7. Fixed "Error: Request timed out." after ~4 chats: root cause was the
   frontend 15s timeout vs. unbounded 20-40s knowledge answers on later
   turns. Fixes (frontend + backend): 90s chat timeout, 60s configurable
   knowledge timeout, knowledge history cap (last 6 turns), `max_tokens`
   bound on both Groq clients. Verified 4-5-turn threads end-to-end.
8. Fixed escalation-interrupt leak: a new question on a thread paused for
   escalation confirmation was consumed as the resume value (echoing the
   stale fallback). `chat_service` now declines the pending escalation
   (`Command(resume=False)`) when the next message isn't an affirmative, then
   processes it as a fresh turn. 2 new unit tests; verified end-to-end both
   the new-question path and the "yes" -> email-collection path.
9. Diagnosed three user-reported failures: (a) HTTP 500 = Groq 429 rate limit
   (200k tokens/day exhausted) crashing the classifier -> graceful degradation
   + short-wait 429 retries + route-level safety net (no more 500s, verified
   live under a real 429); (b) "is Katherine Mah an advisor?" ungrounded =
   empty vector retrieval + extraction not resolving the entity -> deterministic
   `entity_name_fallback` (verified live: Katherine Mah -> holds_role -> Advisor);
   (c) "what is alpinist?" out-of-scope = borderline classifier call -> added
   company-overview wording to the domain definition (needs rate-limit reset to
   verify). Rebuilt containers.
10. Root-caused "doesn't answer any questions" after the rate limit reset: a
    brand-new bug in the fallback wiring. The vector-search node wrote
    `structured_facts` on its own, so under the *hybrid* strategy its parallel
    sibling `structured_lookup` wrote the same channel in one superstep ->
    `InvalidUpdateError: At key 'structured_facts': Can receive only one value
    per step` -> silent error marker -> "Sorry, something went wrong" on every
    hybrid query. Also, the fallback only lived in the vector node, but
    strategy `structured` never ran it; and `match_entity_label`'s
    longest-match heuristic resolved "is Chris Byers the founder of Alpinist
    Studios" (history-rewritten) to the COMPANY, returning 14 company facts and
    no Chris Byers relation -> ungrounded. FIXES: (a) `structured_facts` is now
    an `Annotated[tuple, operator.add]` reducer channel (state.py) so parallel
    writers merge; (b) the fallback now fires in BOTH the structured node (when
    EAV lookup is empty) and the vector node (when vector retrieval is empty);
    (c) `match_entity_label` replaced by `match_entity_labels` — matches EVERY
    named entity ordered by first occurrence (subject first), so both "Chris
    Byers" and "Alpinist Studios" resolve with subject facts first. Verified
    end-to-end: a 4-turn thread (company overview, "elaborate team", "is
    Katherine Mah an advisor", "is Chris Byers the founder") all answered
    grounded, including the previously-failing hybrid + history case. 6 new
    tests (reducer regression proves it fails without the fix). Rebuilt
    containers.
11. Fixed "doesn't answer everything it should" (source-inventory questions):
    "how many links have I ingested?" was misclassified OUT_OF_SCOPE, and
    "how many sources do you have?" was ungrounded because the count lives
    only in `knowledge_source` (never in chunks/EAV). FIXES: (a) domain
    definition now includes ingested links/sources/pages/documents questions
    as in-scope; (b) new `source_inventory_fallback` in `structured_lookup.py`
    — deterministic read of active `knowledge_source` rows producing a count
    fact + one `ingested_link` fact per source — wired into BOTH retrieval
    nodes after the entity-name fallback comes up empty. Verified end-to-end:
    all three count/inventory phrasings answer ("2 links"), plus regressions
    still pass. 5 new tests. Rebuilt containers.
12. Member-roster fallback + the "who are the members" history bug: "who are
    the members" AFTER one prior turn rewrites to "Who are the members of
    Alpinist Studios?" — extraction stochastically returns either
    `relation_type='members'` (EAV lookup returns 0 facts -> empty gate -> old
    fallback chain runs) or `relation_type=None` (general EAV lookup returns
    ~73 company attribute facts, NON-empty -> `if not facts:` gate skipped ->
    roster facts never reached the context -> ungrounded). FIXES: (a) new
    `member_roster_fallback` + `looks_like_member_roster_question` +
    `_member_roster_statement` in `structured_lookup.py` — all Person entities
    + their outgoing relations — wired into BOTH retrieval nodes; (b) the
    structured node's gate is now `not facts OR
    looks_like_member_roster_question(rewritten_text) OR
    looks_like_source_inventory_question(rewritten_text)`, so intent-flavored
    queries run the deterministic fallbacks even when the EAV lookup returned
    non-empty company attributes; merged `(*roster, *named, *inventory, *facts)`
    — roster FIRST so the 20-fact context budget keeps them (all fallback facts
    are confidence 1.0; ranking is a stable sort so insertion order survives).
    Verified live end-to-end: the full 6-question thread answers "who are the
    members" after "what is alpinist studios". New gate regression test in
    `test_nodes_fallback.py`. NOTE: `docker cp` of changed files was LOST when
    the container was recreated (API-key change) — the rebuilt image is the
    source of truth; rebuild (`docker compose up -d --build --no-deps backend
    worker`) after source changes, don't rely on docker cp.
13. Chat latency root-caused and mitigated (gpt-oss-120b -> gpt-oss-20b +
    `reasoning_effort`): "why is it taking so long?" turned out to be TWO
    things. (a) The Groq ORG's per-model daily token budget (200k TPD,
    rolling) — the original API key change was to the SAME org, so the new
    key did NOT reset the exhausted `gpt-oss-120b` budget. Every chat turn
    makes ~4 sequential Groq calls (~3-6k tokens); under sustained use the
    budget exhausts mid-run, calls 429, and the bounded retries (up to 5s
    waits) stretch turns to 20-50s — sometimes erroring into the safe
    fallback. (b) `gpt-oss-120b` and `gpt-oss-20b` are REASONING models; the
    answer prompt is large (~10-11k chars: up to 20 structured facts +
    chunks + history) so a call burns the 1024-token cap on chain-of-thought
    and takes 5-15s even with budget available (same prompt, <1s when small).
    MITIGATIONS (all validated live): switched BOTH Groq clients'
    default model to `openai/gpt-oss-20b` (its OWN TPD budget; verified it
    returns clean extraction JSON + good answers — the org's 120b budget was
    permanently exhausted), and set `reasoning_effort="low"` on every Groq
    chat call (`llm_client.py` `_complete`, `providers.py` Groq `_complete`)
    to cut reasoning latency/tokens. Verified: turn-1 latency ~3-4s and all 6
    target questions answer when the budget is free. Residual slowness on
    later turns / occasional errors is the org token budget being exhausted
    during sustained use — a tier/rotation constraint, not the code. (The
    old `llama-3.x` models are 404 on the current key; `qwen/qwen3.6-27b`
    always emits a "thinking" preamble in `content` on Groq, so it's not
    usable for extraction.) 334 tests still pass.

## 8. Next moves (candidates)

- Confirm the Groq daily token budget reset — upgrades/rotation aside, verify
  the "what is alpinist" classification and 500-fix once budget frees.
- Decide `where is alpinist located?` (add location to domain definition?).
- Consider LANGGRAPH_STRICT_MSGPACK + wiring the REAL groundedness check
  (currently trusts the knowledge `is_grounded` hint).
- COMMIT the uncommitted `feature/updated_ontology` work.
- Capability phrasings ("can alpinist build a custom mobile app") can still be
  ungrounded: the provides facts cover "MVP Development (mobile, web, custom
  apps)" but extraction doesn't map an arbitrary product phrase to the
  `provides` relation, so the fallbacks find nothing and it returns the honest
  ungrounded fallback. Also "if i want to build something can it" occasionally
  gets a clarification question instead of the services answer (classifier
  variance). Both are LLM-routing/extraction variance, not missing data.
