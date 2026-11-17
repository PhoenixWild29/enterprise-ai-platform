# ADR 001: Run on Amazon Bedrock AgentCore with Strands; keep the gates framework-independent

Status: accepted. Date: 2026-11-17.

## Context

The target platform runs on Amazon Bedrock AgentCore with the Strands Agents SDK, over a Snowflake
lakehouse, with Microsoft Entra ID identities. The first version of this repository was built on
LangGraph and FastAPI. Three independent reviews showed that the hard guarantees (read-only SQL,
allow-listed tables and functions, entitlement, row-level security, evidence-only answers) are the
part that matters and the part that is expensive to get right.

## Decision

1. The guarantees live in plain Python (`gates.py`, `platform_tools.py`) with no dependency on an
   agent framework. Any runtime calls the same functions.
2. The AgentCore runtime uses one Strands agent with gated tools instead of a supervisor and
   specialists. Recent models handle a small tool set well in one context, a single agent is cheaper
   and simpler to trace, and none of the guarantees depended on the split.
3. The model never writes the released answer. It returns which evidence answers the question as
   structured output; code assembles the answer from whole passages and runs the citation gate.
4. The LangGraph runtime stays for local development, CI and the evaluation harness, and because it
   has durable pause-and-resume for human review. Both runtimes return the same result shape.
5. Models are called through Bedrock in the account's own VPC. Roles (routing, SQL, synthesis) can use
   different models. Every request has a token ceiling.
6. Memory writes are gated like answers: a policy check, private by default, review before sharing.

## Consequences

- A change of orchestration framework does not reopen the security review.
- Answers are quoted evidence rather than prose. This is deliberate and is the first thing to revisit
  once an entailment check is trusted.
- Two runtimes must be kept in step. The shared result shape and shared tools limit the cost.
- Nothing here has run against live AWS services yet. See the verification table in the README.

## Alternatives considered

- Supervisor and specialists on Strands (agents as tools): more model calls and harder traces for no
  additional guarantee. Worth re-testing with the evaluation harness as tool counts grow.
- Letting the agent write prose and checking it with a second model: a reviewer showed this can release
  an unsupported claim, so it is refused in production.
- Dropping LangGraph: would lose durable human review until AgentCore-side interrupts are wired in.
