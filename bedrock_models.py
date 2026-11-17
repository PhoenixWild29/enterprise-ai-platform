"""bedrock_models.py - Amazon Bedrock model access, model portfolio and cost guardrails.

Models are called through the Bedrock Converse API inside the caller's own
AWS account, so prompts and data never leave it.

  ModelPortfolio   which model serves which role (cheap triage for routing,
                   stronger models for SQL generation and synthesis)
  UsageMeter       token and cost accounting per role, model and principal
  RequestBudget    a per-request token ceiling; exceeding it stops the run
  bedrock_completion   `prompt -> text` function for one role, metered

Configuration:
  AWS_REGION                      region of the Bedrock runtime
  AGENT_MODEL                     default Bedrock model id or inference profile
  MODEL_ROUTER, MODEL_SQL, MODEL_SYNTH, MODEL_VERIFIER   optional per-role overrides
  MODEL_PRICES_JSON               {"<model id>": {"input_per_mtok": 0.0, "output_per_mtok": 0.0}}
  MAX_TOKENS_PER_REQUEST          per-request token ceiling (default 60000)
Prices are configuration, not code: they change, and they differ by region
and by contract. A model with no configured price is metered in tokens only.
"""

import contextvars
import json
import logging
import os
import threading
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("platform.models")

ROLES = ("router", "sql", "synth", "verifier", "judge")
DEFAULT_MAX_TOKENS_PER_REQUEST = 60_000


class BudgetExceeded(RuntimeError):
    """The request used more tokens than its budget allows."""


class ModelPrice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_per_mtok: float = Field(ge=0)
    output_per_mtok: float = Field(ge=0)

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1_000_000


def load_prices(raw: str | None = None) -> dict[str, ModelPrice]:
    raw = raw if raw is not None else os.environ.get("MODEL_PRICES_JSON", "")
    if not raw.strip():
        return {}
    try:
        return {model: ModelPrice.model_validate(price) for model, price in json.loads(raw).items()}
    except (ValueError, AttributeError) as err:
        raise RuntimeError(f"MODEL_PRICES_JSON is invalid: {err}") from err


@dataclass
class RequestBudget:
    """Token usage of one request, checked against a ceiling."""

    max_total_tokens: int = DEFAULT_MAX_TOKENS_PER_REQUEST
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    unpriced_calls: int = 0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def check(self) -> None:
        """Called before each model call: a request over budget makes no further calls."""
        if self.total_tokens >= self.max_total_tokens:
            raise BudgetExceeded(
                f"Request used {self.total_tokens} tokens; the budget is {self.max_total_tokens}."
            )

    def add(self, input_tokens: int, output_tokens: int, cost: float | None) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            if cost is None:
                self.unpriced_calls += 1
            else:
                self.cost_usd += cost

    def summary(self) -> dict[str, Any]:
        return {
            "model_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": None if self.unpriced_calls else round(self.cost_usd, 6),
            "budget_tokens": self.max_total_tokens,
        }


_REQUEST_BUDGET: contextvars.ContextVar[RequestBudget | None] = contextvars.ContextVar("request_budget", default=None)


def current_budget() -> RequestBudget | None:
    return _REQUEST_BUDGET.get()


@contextmanager
def request_budget(max_total_tokens: int | None = None) -> Iterator[RequestBudget]:
    """Scope a token budget to one request. Model calls made inside the block
    count against it, on this thread and in work started with a copy of this
    context (asyncio tasks, LangGraph nodes). A bare threading.Thread does not
    copy the context, so calls made there are not budgeted."""
    ceiling = max_total_tokens or int(os.environ.get("MAX_TOKENS_PER_REQUEST", DEFAULT_MAX_TOKENS_PER_REQUEST))
    budget = RequestBudget(max_total_tokens=ceiling)
    token = _REQUEST_BUDGET.set(budget)
    try:
        yield budget
    finally:
        _REQUEST_BUDGET.reset(token)


class UsageMeter:
    """Process-wide usage totals for cost reporting and forecasting."""

    def __init__(self, prices: dict[str, ModelPrice] | None = None) -> None:
        self.prices = prices if prices is not None else load_prices()
        self._lock = threading.Lock()
        self._totals: dict[tuple[str, str], dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        )

    def record(self, role: str, model_id: str, input_tokens: int, output_tokens: int) -> float | None:
        price = self.prices.get(model_id)
        cost = price.cost(input_tokens, output_tokens) if price else None
        with self._lock:
            row = self._totals[(role, model_id)]
            row["calls"] += 1
            row["input_tokens"] += input_tokens
            row["output_tokens"] += output_tokens
            row["cost_usd"] += cost or 0.0
        if (budget := current_budget()) is not None:
            budget.add(input_tokens, output_tokens, cost)
        return cost

    def report(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"role": role, "model_id": model, "priced": model in self.prices, **{k: (round(v, 6) if k == "cost_usd" else int(v)) for k, v in row.items()}}
                for (role, model), row in sorted(self._totals.items())
            ]


METER = UsageMeter()


@dataclass(frozen=True)
class ModelPortfolio:
    """Which model serves which role. Unset roles use the default model."""

    default: str
    by_role: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "ModelPortfolio":
        default = os.environ.get("AGENT_MODEL", "")
        if not default:
            raise RuntimeError("Set AGENT_MODEL to a Bedrock model id or inference profile.")
        overrides = {
            role: value
            for role, env in (("router", "MODEL_ROUTER"), ("sql", "MODEL_SQL"), ("synth", "MODEL_SYNTH"), ("verifier", "MODEL_VERIFIER"))
            if (value := os.environ.get(env))
        }
        return cls(default=default, by_role=overrides)

    def model_for(self, role: str) -> str:
        return self.by_role.get(role, self.default)


def bedrock_client(region: str | None = None) -> Any:
    import boto3  # imported lazily: optional dependency
    from botocore.config import Config

    return boto3.client(
        "bedrock-runtime",
        region_name=region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
        config=Config(retries={"max_attempts": 4, "mode": "adaptive"}, read_timeout=120),
    )


def bedrock_completion(
    model_id: str,
    role: str = "agent",
    client: Any = None,
    meter: UsageMeter = METER,
    max_output_tokens: int = 1024,
) -> Callable[[str], str]:
    """`prompt -> text` through the Bedrock Converse API, metered and budgeted.
    The client is created on first use so configuration errors surface at
    startup and credentials are only needed when a model is actually called."""
    holder = {"client": client}

    def complete(prompt: str) -> str:
        if (budget := current_budget()) is not None:
            budget.check()
        if holder["client"] is None:
            holder["client"] = bedrock_client()
        response = holder["client"].converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_output_tokens, "temperature": 0},
        )
        usage = response.get("usage", {})
        meter.record(role, model_id, int(usage.get("inputTokens", 0)), int(usage.get("outputTokens", 0)))
        blocks = response.get("output", {}).get("message", {}).get("content", [])
        return "".join(block.get("text", "") for block in blocks)

    return complete


def portfolio_completions(portfolio: ModelPortfolio, client: Any = None, meter: UsageMeter = METER) -> dict[str, Callable[[str], str]]:
    """One metered completion function per role, sharing one client."""
    holder = {"client": client}

    def lazy_client() -> Any:
        if holder["client"] is None:
            holder["client"] = bedrock_client()
        return holder["client"]

    class _Lazy:
        def converse(self, **kwargs: Any) -> Any:
            return lazy_client().converse(**kwargs)

    shared = _Lazy()
    return {role: bedrock_completion(portfolio.model_for(role), role=role, client=shared, meter=meter) for role in ROLES}
