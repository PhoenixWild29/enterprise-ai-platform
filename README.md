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
| `auth.py` | Entra ID OBO user path, M2M service-principal path, authorization, row-level security |
| `graph.py` | LangGraph `StateGraph`, injectable agent dependencies |
| `eval_harness.py` | Tier 1 gold-benchmark regression, Tier 2 judge with 2x false-negative penalty |
| `main.py` | FastAPI service |

## Run

```bash
pip install -r requirements.txt
pytest -q                      # unit, graph and API tests
python eval_harness.py         # both evaluation tiers
uvicorn main:app --reload      # API at http://127.0.0.1:8000/docs
```

```bash
TOKEN=$(curl -s -X POST localhost:8000/v1/dev/token -H 'content-type: application/json' \
  -d '{"upn":"dana.ops@contoso.com","role":"Lakehouse.Operator"}' | python -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')
curl -s -X POST localhost:8000/v1/query -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"query":"List the latest telemetry readings"}'
```

## What is simulated

- **Entra ID**: in-process issuer, HS256 tokens, dev-only default secrets. Production needs RS256/JWKS via MSAL.
- **Models**: router, retriever, SQL generator and synthesizer are deterministic stand-ins behind `AgentDeps`.
- **Snowflake**: the executor returns synthetic rows; the catalog is generated.
- **Checkpointer**: in-memory, so paused reviews do not survive a restart.
- **LLM judge**: `--judge llm` needs `anthropic`, `ANTHROPIC_API_KEY` and `JUDGE_MODEL`; CI uses the deterministic heuristic judge.
