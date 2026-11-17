# Enterprise AI Platform (prototype)

Multi-agent LangGraph workflow over a governed Snowflake lakehouse (500-table
catalog: SCADA telemetry, maintenance, finance) with fail-closed validation
gates. Python 3.12, LangGraph, Pydantic v2, sqlglot, FastAPI.

## Flow

```
START -> auth_gate -> supervisor_router
  RAG         rag_retriever -> thin_set_gate -> synthesizer
  TEXT-TO-SQL text_to_sql -> sql_ast_gate -> payload_safety_gate -> authz_gate -> sql_executor -> synthesizer
  DIAGNOSTIC  diagnostic_reasoner -> thin_set_gate -> (SQL chain) -> synthesizer
synthesizer -> citation_gate -> respond | hitl_review | refusal -> END
```

Any gate that does not pass sets `gate_status` to `REFUSED` and routes to an
output node. A citation failure on the user path pauses for human review.

## Modules

| File | Purpose |
| --- | --- |
| `schemas.py` | `AgentState`, catalog, tool payload, MCP tool and OpenTelemetry span models |
| `gates.py` | Thin-set refusal, SQL AST parser, payload and safety, verbatim citation gates |
| `auth.py` | Identity-provider interface, OBO user path, M2M path, authorization, row-level security |
| `entra.py` | Microsoft Entra ID provider: RS256/JWKS validation, MSAL token exchange |
| `graph.py` | LangGraph `StateGraph`, injectable agent dependencies |
| `llm_agents.py` | Model-backed router, Text-to-SQL (schema retrieval, gate-driven repair) and synthesizer |
| `persistence.py` | Checkpointer selection: SQLite, Postgres or memory |
| `eval_harness.py` | Tier 1 gold-benchmark regression, Tier 2 judge with 2x false-negative penalty |
| `main.py` | FastAPI service |

## Run

```bash
pip install -r requirements.txt
pytest -q                      # unit, graph, API, provider and persistence tests
python eval_harness.py         # both evaluation tiers
uvicorn main:app --reload      # API at http://127.0.0.1:8000/docs
```

```bash
TOKEN=$(curl -s -X POST localhost:8000/v1/dev/token -H 'content-type: application/json' \
  -d '{"upn":"dana.ops@contoso.com","role":"Lakehouse.Operator"}' | python -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')
curl -s -X POST localhost:8000/v1/query -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"query":"List the latest telemetry readings"}'
```

## Configuration

| Variable | Values | Notes |
| --- | --- | --- |
| `APP_ENV` | `dev` (default), `production` | Production refuses the simulated identity provider and the memory checkpointer at startup |
| `IDENTITY_PROVIDER` | `simulated` (default), `entra` | `entra` needs `ENTRA_TENANT_ID`, `ENTRA_CLIENT_ID`, `ENTRA_CLIENT_SECRET`, `SNOWFLAKE_SCOPE_PREFIX`; optional `ENTRA_API_AUDIENCE`, `SERVICE_PRINCIPALS_JSON` |
| `CHECKPOINT_BACKEND` | `sqlite` (default), `postgres`, `memory` | `CHECKPOINT_SQLITE_PATH` (default `.data/checkpoints.sqlite`) or `CHECKPOINT_DSN` |
| `AGENT_BACKEND` | `simulated` (default), `llm` | `llm` needs `pip install anthropic`, `ANTHROPIC_API_KEY`, `AGENT_MODEL` |
| `JUDGE_MODEL` | model id | For `python eval_harness.py --judge llm`; add `--agents llm` to evaluate the model-backed agents |

## Verification status

| Component | Status |
| --- | --- |
| Gates, graph, API, eval harness | Tested in CI |
| SQLite and Postgres checkpointers | Tested in CI, including resume after a simulated restart |
| Entra provider | RS256 validation tested with a generated key pair; the JWKS download and MSAL calls are faked. Not yet run against a real tenant |
| Model-backed agents and LLM judge | Logic tested with scripted completions. Not yet run against a live model |
| Snowflake | Not connected. The executor returns synthetic rows and the catalog is generated |
| Retriever | Keyword match over a six-passage sample corpus. No vector store |
