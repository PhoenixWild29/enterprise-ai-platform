"""Bedrock Converse adapter, model portfolio and budgets, against a stubbed boto3 client."""
import boto3
import pytest
from botocore.stub import Stubber

from bedrock_models import (
    BudgetExceeded, ModelPortfolio, ModelPrice, UsageMeter, bedrock_completion, load_prices, portfolio_completions, request_budget,
)
from llm_agents import LLMRouter
from schemas import Route

MODEL = "us.example.model-v1:0"


def _client():
    return boto3.client("bedrock-runtime", region_name="us-east-1", aws_access_key_id="x", aws_secret_access_key="y")


def _reply(text, tokens_in=120, tokens_out=30):
    return {"output": {"message": {"role": "assistant", "content": [{"text": text}]}}, "stopReason": "end_turn",
            "usage": {"inputTokens": tokens_in, "outputTokens": tokens_out, "totalTokens": tokens_in + tokens_out},
            "metrics": {"latencyMs": 10}}


def _expect(prompt, model=MODEL):
    return {"modelId": model, "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": 1024, "temperature": 0}}


def test_converse_request_shape_metering_and_cost():
    client, meter = _client(), UsageMeter({MODEL: ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0)})
    with Stubber(client) as stub:
        stub.add_response("converse", _reply("sql"), _expect("classify this"))
        complete = bedrock_completion(MODEL, role="router", client=client, meter=meter)
        with request_budget(10_000) as budget:
            assert complete("classify this") == "sql"
        stub.assert_no_pending_responses()
    assert budget.summary() == {"model_calls": 1, "input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
                                "estimated_cost_usd": round((120 * 3 + 30 * 15) / 1e6, 6), "budget_tokens": 10_000}
    assert meter.report() == [{"role": "router", "model_id": MODEL, "priced": True, "calls": 1, "input_tokens": 120,
                               "output_tokens": 30, "cost_usd": round((120 * 3 + 30 * 15) / 1e6, 6)}]


def test_unpriced_model_reports_tokens_but_no_invented_cost():
    client, meter = _client(), UsageMeter({})
    with Stubber(client) as stub:
        stub.add_response("converse", _reply("ok"), _expect("p"))
        with request_budget() as budget:
            bedrock_completion(MODEL, client=client, meter=meter)("p")
    assert budget.summary()["estimated_cost_usd"] is None and budget.total_tokens == 150


def test_budget_stops_further_model_calls_and_agents_fail_closed():
    client, meter = _client(), UsageMeter({})
    with Stubber(client) as stub:
        stub.add_response("converse", _reply("rag", tokens_in=900, tokens_out=200), _expect("first"))
        complete = bedrock_completion(MODEL, client=client, meter=meter)
        with request_budget(1_000):
            complete("first")
            with pytest.raises(BudgetExceeded):
                complete("second")  # no request is sent: the stub has no second response queued
            assert LLMRouter(complete)("where is the pump manual?") is Route.REFUSE
        stub.assert_no_pending_responses()


def test_portfolio_routes_roles_to_models(monkeypatch):
    for name in ("AGENT_MODEL", "MODEL_ROUTER", "MODEL_SQL", "MODEL_SYNTH", "MODEL_VERIFIER"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="AGENT_MODEL"):
        ModelPortfolio.from_env()
    monkeypatch.setenv("AGENT_MODEL", "strong-model")
    monkeypatch.setenv("MODEL_ROUTER", "cheap-model")
    portfolio = ModelPortfolio.from_env()
    assert portfolio.model_for("router") == "cheap-model" and portfolio.model_for("sql") == "strong-model"

    client, meter = _client(), UsageMeter({})
    with Stubber(client) as stub:
        stub.add_response("converse", _reply("sql"), _expect("route me", "cheap-model"))
        stub.add_response("converse", _reply("SELECT 1"), _expect("write sql", "strong-model"))
        completions = portfolio_completions(portfolio, client=client, meter=meter)
        assert completions["router"]("route me") == "sql" and completions["sql"]("write sql") == "SELECT 1"
    assert [(r["role"], r["model_id"]) for r in meter.report()] == [("router", "cheap-model"), ("sql", "strong-model")]


def test_price_configuration(monkeypatch):
    assert load_prices("") == {}
    assert load_prices('{"m": {"input_per_mtok": 1, "output_per_mtok": 2}}')["m"].cost(1_000_000, 500_000) == 2.0
    with pytest.raises(RuntimeError, match="MODEL_PRICES_JSON"):
        load_prices('{"m": {"input_per_mtok": -1, "output_per_mtok": 2}}')
    with pytest.raises(RuntimeError):
        load_prices("not json")
