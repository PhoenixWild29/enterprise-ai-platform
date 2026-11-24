# Enterprise AI Platform (prototype)

Multi-agent LangGraph workflow over a governed Snowflake lakehouse (500-table
catalog: SCADA telemetry, maintenance, finance) with fail-closed validation
gates. Python 3.12, LangGraph, Pydantic v2, sqlglot, FastAPI.

## Two runtimes, one set of gates

| Runtime | Entry point | Orchestration | Use |
| --- | --- | --- | --- |
| Amazon Bedrock AgentCore | `agentcore_app.py` (`POST /invocations`) | One Strands agent with gated tools (`strands_agent.py`) | The target deployment |
| FastAPI | `main.py` | LangGraph supervisor and specialists (`graph.py`) | Local development, CI, the evaluation harness, durable human review |

The guarantees do not live in either orchestrator. They live in `gates.py` and in the tools in
`platform_tools.py`, which both runtimes call. The MCP server (`mcp_server.py`) exposes the same tools.

### How it maps to AgentCore

| AgentCore component | In this repository |
| --- | --- |
| Runtime | `agentcore_app.py`, `Dockerfile`, `infra/terraform` |
| Gateway (MCP tools) | `mcp_server.py`: `find_tables`, `search_manuals`, `run_sql`, each run as the caller |
| Identity | `agentcore_identity.py`: on-behalf-of exchange for users; scheduled jobs present their own app-only token and get a machine-flow token for their one role |
| Memory | `memory.py`: session, user, persona and organisation scopes with a write policy and review; `AgentCoreMemoryStore` |
| Observability | `telemetry.py`: both runtimes emit one trace per run from the audit trail, over OTLP when an endpoint is set. Spans carry decisions and counts, never query text, SQL, gate messages or a person's identity |
| Models | `bedrock_models.py`: Bedrock Converse, per-role model portfolio, token budgets, cost metering |

## Flow (LangGraph runtime)

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
| `platform_tools.py` | The gated tools shared by the Strands agent and the MCP server; per-request evidence ledger |
| `strands_agent.py` | Strands agent: gated tools, security halt hook, structured evidence output, extractive answer |
| `agentcore_app.py` | AgentCore Runtime entrypoint for user and scheduled invocations |
| `agentcore_identity.py` | Identity provider backed by AgentCore Identity |
| `bedrock_models.py` | Bedrock model access, model portfolio by role, request budgets, usage and cost metering |
| `memory.py` | Governed memory: scopes, write policy, review of shared memory, retention |
| `mcp_server.py` | MCP server exposing the gated tools with bearer-token authentication |
| `telemetry.py` | OpenTelemetry export of the audit trail |
| `anomaly.py` | Classical anomaly detection (robust z-score) for sensor readings |
| `graph_mail.py` | Mailbox search tools over Microsoft Graph: server-side filters, sequential paging with a caller-bound cursor, counting in code, one message read on request |
| `quality_eval.py` | Text-to-SQL quality evaluation: 46 questions with known answers, a code grader that compares rows, both runtimes, and an audit of the evaluation itself |
| `local_warehouse.py` | In-memory DuckDB seeded with deterministic data, so governed SQL really executes and a result can be graded |
| `agentcore_shared_memory.py` | Shared memory on AgentCore Memory: per-person extraction strategies, shared namespaces written only by an approved promotion, screened recall |
| `snowflake_backend.py` | Snowflake executor (caller's own OAuth token), catalog loader from Snowflake metadata |
| `retrieval.py` | Vector index, offline embedder, Snowflake Cortex vector retriever |
| `runtime.py` | Assembles agents, warehouse, retriever and catalog from the environment |
| `eval_harness.py` | Tier 1 gold-benchmark regression, Tier 2 judge with 2x false-negative penalty |
| `main.py` | FastAPI service |

## Run

```bash
pip install -r requirements-dev.txt
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
| `APP_ENV` | `dev` (default), `production` | Production refuses every development stand-in at startup: identity provider, checkpointer, agents, warehouse, retriever and catalog |
| `IDENTITY_PROVIDER` | `simulated` (default), `entra`, `agentcore` | `entra` needs `ENTRA_TENANT_ID`, `ENTRA_CLIENT_ID`, `ENTRA_CLIENT_SECRET`, `SNOWFLAKE_SCOPE_PREFIX`; optional `ENTRA_API_AUDIENCE`, `ENTRA_ALLOWED_CLIENT_IDS`, `SERVICE_PRINCIPALS_JSON`. Tokens must be v2.0 with the `upn` optional claim |
| `CHECKPOINT_BACKEND` | `sqlite` (default), `postgres`, `memory` | `CHECKPOINT_SQLITE_PATH` (default `.data/checkpoints.sqlite`) or `CHECKPOINT_DSN` |
| `AGENT_BACKEND` | `simulated` (default), `llm` | `llm` needs `AGENT_MODEL` (a Bedrock model id or inference profile) |
| `MODEL_PROVIDER` | `bedrock` (default), `anthropic` | Production requires `bedrock`. Optional per-role models: `MODEL_ROUTER`, `MODEL_SQL`, `MODEL_SYNTH`, `MODEL_VERIFIER` |
| `MAX_TOKENS_PER_REQUEST` | integer (default 60000) | Token ceiling per request; a request over budget makes no further model calls |
| `MODEL_PRICES_JSON` | JSON | Prices per million tokens by model id, for cost reporting. Unpriced models are metered in tokens only |
| `WAREHOUSE_BACKEND` | `simulated` (default), `snowflake` | `snowflake` needs `pip install snowflake-connector-python`, `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_WAREHOUSE`; optional `SNOWFLAKE_DATABASE` |
| `RETRIEVER_BACKEND` | `keyword` (default), `vector`, `hybrid`, `snowflake` | `vector`, `hybrid`: `CORPUS_PATH` (JSONL). `snowflake`: `PASSAGES_TABLE`, `SNOWFLAKE_CONNECTION_NAME` |
| `CATALOG_PATH` | file path | Catalog JSON from `python snowflake_backend.py export-catalog --overlay governance.json --out catalog.json` |
| `SYNTHESIS_MODE` | `extractive` (default), `abstractive` | Extractive releases only verbatim source text. Abstractive releases model-written claims after a literal check and a verifier-model check; failures go to human review |
| `AGENTCORE_OBO_PROVIDER`, `AGENTCORE_M2M_PROVIDER`, `AGENTCORE_MEMORY_ID` | names and id | AgentCore credential providers and memory, for `IDENTITY_PROVIDER=agentcore` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | URL | Where to export traces; nothing is exported when unset |
| `JUDGE_MODEL` | model id | For `python eval_harness.py --judge llm`; add `--agents llm` to evaluate the model-backed agents |

## Security properties

- **SQL**: one read-only SELECT, executed as re-serialised from the validated syntax tree; tables allow-listed by the catalog, with CTE names resolved in scope; functions allow-listed in `gates.py` (system, table and session-context functions are refused); no time travel, sequences, bind or session variables, pivots, sampling, recursive CTEs or quoted identifiers; joins on declared keys; time bounds and LIMIT caps enforced.
- **Answers**: every released line carries a verbatim quote from retrieved evidence. With the default extractive synthesizer no model-written sentence is released, and each chosen passage is released whole, so no cut can drop a number, a condition or a negation. Chunk the corpus so each passage stands on its own. Abstractive mode is refused in production.
- **Evidence**: passages with no data domain are shown to no one; retrieval failures are refusals.
- **Identity**: RS256 only, issuer, audience, tenant and expiry checked; `upn` required; service-principal secrets are checked by Entra on every call.
- **Tokens**: access tokens live only in an in-process vault, bound to the principal they were issued to. State, checkpoints, logs and responses carry a fingerprint.
- **Checkpoints**: a run's state is deleted when it finishes or crashes; only runs waiting on a reviewer persist. The SQLite file is owner-only, with secure delete on and the write-ahead log truncated after each delete. Encrypt the store at rest.
- **Errors**: warehouse and connector messages are logged server-side and not returned to callers.
- **Query results**: rows are released as evidence, so a SELECT list is an allow-list of shapes: a column, `*`, COUNT/SUM/AVG/MIN/MAX of a column, ROUND of those, and DATE_TRUNC of a column. Output names are derived from the shape (`avg_value`, `n`); an author cannot choose one. Every computed output must be named, because a warehouse names an unnamed expression after its text, table alias included. Extra arguments on those functions, GROUPING SETS, ROLLUP and CUBE, non-ASCII names, set operations, column lists on CTEs and subqueries, `* RENAME`, QUALIFY, DISTINCT ON, parenthesised joins, non-standard string forms and LIMIT anywhere but the outermost query are refused. The policy is applied twice: to the query as written and to the re-serialised SQL that will actually execute, because the SQL generator can rewrite a construct into a different one. The aim is that a query cannot hand back text or a number its author wrote. As a second check on what actually comes back, a result is not released if it has a column that is neither a catalog column of a table the query reads nor one of the query's own derived names. What a count or aggregate means still depends on the query's filters, which is why the executed SQL is returned with the answer.
- **Scheduled jobs**: a job is whoever its verified app-only token says it is, and must be registered for the DAG it names. Nothing in a request body confers an identity.
- **Relevance**: on the Strands runtime the agent picks its own search strings, so a document is released only if it also clears the thin-set gate for the question the caller actually asked.
- **Memory**: a write policy screens out text that looks like a secret, an instruction to the agent, or query results (pattern-based, see limitations). Entries are capped per person. The agent cannot write memory once any document or data is in its context. People write only to their own memory; shared memory is a proposal until a different admin approves it. Recalled memory is placed in prompts as escaped data.
- **Cost**: every request has a token ceiling; usage is metered by role and model.
- **Environment**: any `APP_ENV` other than `dev`, `development`, `local` or `test` gets production rules.

## Known limitations

- The Strands runtime reports HITL_REVIEW but has no pause-and-resume store; durable review exists only in the FastAPI runtime.
- Shared (persona and organisation) memory is held in process; only user memory persists to AgentCore Memory.
- The memory write policy is pattern-based. It blocks obvious secrets and instructions, including zero-width characters and words that mix two alphabets, but a reworded instruction or a planted "fact" gets through. What limits the damage is that memory only ever reaches its author's own prompt, as escaped data, and cannot change a gate.
- The projection rule is strict: arithmetic, CASE, window functions, casts, unit conversions and constants in a SELECT list are refused, so some legitimate analytical queries cannot be expressed yet.
- The platform's own traces withhold query text, SQL and identities. The Strands SDK's built-in tracing does not: if OpenTelemetry auto-instrumentation is switched on for the container, its spans carry prompts, SQL and user names. Leave it off, or redact at the collector.
- A review resume gets its own token ceiling, so a run that pauses for review can spend up to twice the per-request limit.
- Joins between two tables that share a foreign key (including a table joined to itself) pass the key check, so SUM or COUNT over one can be inflated by row fan-out. The values still come from table data.
- Derived output names (`n`, `day`, `max_value`) assume the catalog has no real column with the same name. The demo catalog has none; a catalog loaded from Snowflake is not checked for this yet.
- Selected column names are not resolved against the catalog; an unknown name fails in the warehouse rather than at the gate.
- A mail search's next links must stay on exactly the path the search started on. If Graph returns a differently spelled path for a folder (for example `mailFolders('inbox')`), inbox and sent searches would stop at the second page with a refusal. This fails loudly, not silently, and needs checking in a tenant.
- The mail tools are standalone: they are not yet registered on the Strands agent, and mail is not citable evidence in an answer. Cursors and message handles are held per process.
- Shared memory promotion is wired to the review step (`MemoryService(publisher=...)`) but the AgentCore entrypoint does not use it yet. Provenance (author, approver) is logged and kept on the platform's own record, joined by the published id; it is not yet written to the AgentCore record's metadata. That platform record, with its published id, is held in process today, so after a restart a published entry can no longer be removed through `forget`; a durable store for shared proposals is needed before this is used for real. `promote` checks that a record is consistent with an approval, not that the approval happened: only `MemoryService.review` does that, so nothing else should be able to call it or write `/shared/`. The IAM policy that makes a namespace an access boundary is described in the notes for CNX, not in the Terraform.
- The quality evaluation's questions and data are invented, 46 cases give wide intervals (roughly plus or minus 12 points), and a real model is not wired in (`--model llm` exits with a message). The local warehouse is DuckDB, not Snowflake: the SQL is translated, so a query that behaves differently on Snowflake would not be caught here.
- Relevance is checked for documents only. Whether returned rows answer the question is not checked on either runtime.
- The MCP server accepts any valid token for the platform API; it has no audience of its own yet, so a token for one is good for the other.
- Scheduled jobs on AgentCore need the optional `idtyp` claim enabled on the Entra app registration; without it they are refused.
- The token ceiling on the Strands runtime relies on the model reporting usage. A model that reports none is bounded only by the turn limit.
- The container image is not pinned by digest and has no dependency lock file.
- No prompt caching and no automatic fallback between models.
- Review decisions are serialised per run inside one API process. Run a single replica, or add a database lock, before scaling out.
- An approved answer is returned to the reviewer's call; the original requester has no endpoint to fetch it.
- The token vault is per process, so a run resumed on another replica cannot reach the warehouse and is refused.
- The SQL policy is an allow-list over what sqlglot parses. It has been attacked by three review passes, not formally verified, and has never been run against a real Snowflake account.

## Two evaluations, two questions

| | `eval_harness.py` | `quality_eval.py` |
| --- | --- | --- |
| Question | Do the gates hold? | How often is the answer right? |
| Expected score | 100%, always | Well below 100%, so an improvement can show |
| Cases | Mostly attacks a reviewer found | Questions with a known answer, in three tiers, plus requests that must be refused |
| Grader | Gate outcome and node path | The rows returned, compared in code with the reference rows |
| Use | Regression gate in CI | Comparing models, prompts and agent designs |

`python quality_eval.py` runs both runtimes on the same questions against a local warehouse and writes `reports/quality_eval.html`.
Both are graded on the same thing: the rows of the last statement that actually ran, and only if the platform released an answer.
A request that must be refused passes only if no query ran at all.

Before reporting a score it audits itself:

- every reference answer has rows, reads the table and the time window its question names, and is not cut short by a `LIMIT`
- the reference answers match a fingerprint recorded with the cases, so the seeded data is the same on every machine
- every planted mistake (wrong table, missing condition, wrong aggregate, wrong window, wrong order) grades as wrong, and the right
  answer still passes with its rows shuffled and its columns renamed
- no question or reference query appears in any file a model reads
- every case gets the same verdict on a second run
- the best score leaves headroom, and a system that answers nothing is called out
- a stronger model scores higher, judged case by case on the same questions
- there are held-out questions in every tier

How answers are read: every answer is over the rows the person asking is entitled to see; "the last N days" means the N times 24 hours
before now; order is graded only when the question asks for one; extra columns are allowed, missing ones are not; numbers match within
half a cent. A real model has to be told the time-window convention or it will be marked wrong.

With 46 cases the 95% interval on a score is roughly plus or minus 12 points, and on a single tier of 12 to 14 questions it is 20 or
more, so each score is printed with its interval.

**The models in it today are scripted stand-ins**, `example-small` and `example-large`. They know the reference queries and make planned
mistakes, so the pipeline, the audit and the report can be built and inspected first. Their scores describe the stand-ins, not any model,
and because both runtimes replay the same script they tie by construction. The numbers mean nothing until a real model is connected.

What the audit cannot tell you: whether the questions are ones people really ask (they are invented, and so is the data), whether a
question has a second defensible reading it has not thought of, and whether DuckDB and Snowflake agree on every function a real model
might use. The planted mistakes are five kinds of error, not every way a query can be wrong.

## Verification status

| Component | Status |
| --- | --- |
| Gates, graph, API, eval harness | Tested in CI |
| SQLite and Postgres checkpointers | Tested in CI, including resume after a simulated restart |
| Strands agent and AgentCore entrypoint | Driven in CI by a scripted model through the real Strands loop and the real AgentCore app, for both the user and the scheduled-job path (with the simulated identity provider). Not yet deployed to AgentCore or run against a Bedrock model |
| Bedrock adapter | Request shape, metering and budgets tested against a stubbed boto3 client. Not yet called against Bedrock |
| AgentCore Identity and Memory adapters | Call shapes tested against fakes built from the SDK's signatures. Not yet run against the services |
| MCP server | Tools exercised through the MCP SDK in-process, including authentication. Not yet placed behind AgentCore Gateway |
| Mail tools | Paging, throttling, cursors and refusals tested against a fake Graph service. Not yet run against Microsoft Graph or a real mailbox. The text-search syntax needs checking in a tenant, and the AgentCore Identity token provider for Graph is untested |
| AgentCore shared memory | Request shapes validated against the service model shipped with boto3. Not yet run against AgentCore Memory. Reads use the exact-match `namespace` parameter (not the hierarchical `namespacePath`), and recall also checks each record's namespace itself |
| Quality evaluation | The warehouse, grader, audit and report are tested: every planted wrong query must grade as wrong, and the audit is tested against planted faults in the labels, the data, the prompts and the split. It has only been run with scripted stand-in models and invented questions on DuckDB; no real model has been scored, so it has not yet answered the single-agent versus supervisor question |
| Terraform | Written from the AWS provider documentation. Never validated, planned or applied. IAM action names and resource ARNs for AgentCore Identity are unverified |
| Entra provider | RS256 validation tested with a generated key pair; the JWKS download and MSAL calls are faked. Not yet run against a real tenant |
| Model-backed agents and LLM judge | Logic tested with scripted completions. Not yet run against a live model |
| Snowflake executor and catalog loader | Logic tested against a fake connection (token handling, row mapping, key discovery, fail-closed paths). Not yet run against a real account |
| Vector retriever | In-process index and offline embedder tested end to end. The Snowflake Cortex retriever is tested against a fake connection only |
| Thin-set threshold | 0.4 is calibrated for neither the offline embedder nor Cortex embeddings; recalibrate per embedder before relying on it |
