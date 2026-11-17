"""AgentCore Runtime entrypoint and AgentCore Identity provider, with the AWS calls faked."""
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from starlette.testclient import TestClient

import agentcore_app
from agentcore_identity import AgentCoreIdentityProvider
from auth import IDP, TELEMETRY_SP_CLIENT_ID, TELEMETRY_SP_SECRET, AuthError, AuthPath, ServicePrincipal, authenticate_service, authenticate_user
from entra import EntraIDProvider
from graph import AgentDeps
from scripted_model import ScriptedModel

ASK = "What is the bearing temperature alarm limit for pump P-101?"
TURNS = [{"tool": ("search_manuals", {"query": "bearing temperature alarm limit pump P-101"})},
         {"tool": ("EvidenceAnswer", {"evidence": [{"passage_id": "sop-pump-014", "quote": "alarm limit is 85 C"}]})}, {"text": "done"}]


@pytest.fixture
def client():
    holder = {}

    class PerRequestModel(ScriptedModel):
        def __init__(self):
            super().__init__(TURNS)

        async def stream(self, *args, **kwargs):
            if not self.turns:
                self.turns = list(TURNS)
            async for event in super().stream(*args, **kwargs):
                yield event

    holder["model"] = PerRequestModel()
    agentcore_app.configure(AgentDeps(), holder["model"], idp=IDP)
    yield TestClient(agentcore_app.app)
    agentcore_app._PLATFORM = None


def _bearer(upn="dana.ops@contoso.com", role="Lakehouse.Operator"):
    return {"Authorization": f"Bearer {IDP.issue_user_token(upn, 'oid-1', [role])}"}


def test_user_invocation_runs_on_behalf_of_the_caller(client):
    r = client.post("/invocations", json={"prompt": ASK}, headers=_bearer() | {"X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "session-0123456789012345678901234567890"})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "PASSED" and body["citations"][0]["passage_id"] == "sop-pump-014"
    assert "85 C" in body["answer"] and body["usage"]["total_tokens"] > 0 and body["refusal_reason"] is None
    assert client.get("/ping").status_code == 200


def test_invocation_without_identity_or_with_bad_input_is_refused(client):
    for headers in ({}, {"Authorization": "Bearer not.a.jwt"}, {"Authorization": "Basic abc"}):
        body = client.post("/invocations", json={"prompt": ASK}, headers=headers).json()
        assert body["status"] == "REFUSED" and "auth_gate" in body["refusal_reason"] and body["citations"] == []
    for payload in ({}, {"prompt": ""}, {"prompt": "x" * 5000}, {"prompt": ASK, "tools": ["shell"]}):
        body = client.post("/invocations", json=payload, headers=_bearer()).json()
        assert body["status"] == "REFUSED" and "[request]" in body["refusal_reason"]


def test_scheduled_job_invocation_uses_the_service_identity(client):
    job = {"client_id": TELEMETRY_SP_CLIENT_ID, "client_secret": TELEMETRY_SP_SECRET,
           "airflow_dag_id": "nightly_telemetry_rollup", "airflow_dag_run_id": "scheduled__2026-11-17"}
    body = client.post("/invocations", json={"prompt": ASK, "job": job}).json()
    assert body["status"] == "REFUSED"  # the telemetry role is not entitled to maintenance manuals: thin evidence
    body = client.post("/invocations", json={"prompt": ASK, "job": job | {"airflow_dag_id": "rogue_dag"}}).json()
    assert body["status"] == "REFUSED" and "not permitted" in body["refusal_reason"]


# ---- AgentCore Identity provider ------------------------------------------------

TENANT, CLIENT = "tenant-123", "platform-client"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
SP = ServicePrincipal(client_id="sp-rollup", service_account_id="svc-telemetry-rollup@contoso",
                      execution_role="SVC_TELEMETRY_ROLLUP_ROLE", allowed_dag_ids=frozenset({"nightly_telemetry_rollup"}))


class FakeIdentityClient:
    def __init__(self, token="snowflake-token", fail=False):
        self.token, self.fail, self.calls = token, fail, []

    async def get_token(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("AccessDenied: tenant internals")
        return self.token


def _user_token(**overrides):
    now = int(time.time())
    claims = {"iss": f"https://login.microsoftonline.com/{TENANT}/v2.0", "aud": f"api://{CLIENT}", "tid": TENANT, "iat": now,
              "exp": now + 600, "oid": "oid-9", "upn": "dana.ops@contoso.com", "roles": ["Lakehouse.Operator"], "scp": "access_as_user"} | overrides
    return jwt.encode(claims, KEY, algorithm="RS256")


def _provider(identity_client, workload_token="workload-token"):
    validator = EntraIDProvider(TENANT, CLIENT, "unused", "api://snowflake-app", key_resolver=lambda t: KEY.public_key(),
                                msal_app_factory=lambda a, b: None)
    return AgentCoreIdentityProvider(validator, "snowflake-obo", "snowflake-m2m", "api://snowflake-app/", [SP],
                                     identity_client=identity_client, workload_token=lambda: workload_token)


def test_agentcore_identity_on_behalf_of_and_machine_flows():
    fake = FakeIdentityClient()
    ctx = authenticate_user(_user_token(), idp=_provider(fake))
    assert ctx.auth_path is AuthPath.USER_OBO and ctx.snowflake_role == "OPS_ANALYST" and ctx.principal == "dana.ops@contoso.com"
    assert fake.calls[0] == {"provider_name": "snowflake-obo", "scopes": ["api://snowflake-app/session:role:OPS_ANALYST"],
                             "agent_identity_token": "workload-token", "auth_flow": "ON_BEHALF_OF_TOKEN_EXCHANGE"}
    svc = authenticate_service("sp-rollup", "", "nightly_telemetry_rollup", "scheduled__1", idp=_provider(fake))
    assert svc.auth_path is AuthPath.M2M and fake.calls[1]["auth_flow"] == "M2M" and fake.calls[1]["provider_name"] == "snowflake-m2m"
    assert "snowflake-token" not in repr(ctx) + repr(svc)


def test_agentcore_identity_fails_closed():
    with pytest.raises(AuthError, match="no AgentCore workload access token"):
        authenticate_user(_user_token(), idp=_provider(FakeIdentityClient(), workload_token=None))
    with pytest.raises(AuthError) as err:
        authenticate_user(_user_token(), idp=_provider(FakeIdentityClient(fail=True)))
    assert "RuntimeError" in str(err.value) and "tenant internals" not in str(err.value)
    with pytest.raises(AuthError, match="no token"):
        authenticate_user(_user_token(), idp=_provider(FakeIdentityClient(token="")))
    with pytest.raises(AuthError, match="audience"):
        authenticate_user(_user_token(aud="api://other"), idp=_provider(FakeIdentityClient()))
    with pytest.raises(AuthError):
        authenticate_service("unknown-sp", "", "nightly_telemetry_rollup", "r", idp=_provider(FakeIdentityClient()))
    with pytest.raises(RuntimeError, match="AGENTCORE_OBO_PROVIDER"):
        AgentCoreIdentityProvider(None, "", "m2m", "api://x")


def test_token_expiry_is_bounded():
    from agentcore_identity import _unverified_expiry
    now = time.time()
    short = jwt.encode({"exp": int(now) + 120}, "k" * 32, algorithm="HS256")
    assert abs(_unverified_expiry(short).timestamp() - (int(now) + 120)) < 2
    long_lived = jwt.encode({"exp": int(now) + 86_400}, "k" * 32, algorithm="HS256")
    assert _unverified_expiry(long_lived).timestamp() <= now + 3601
    assert _unverified_expiry("opaque-token").timestamp() <= now + 901
