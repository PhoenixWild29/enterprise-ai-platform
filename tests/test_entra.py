"""Entra ID provider: real RS256 validation with a locally generated key pair;
only the calls to Microsoft (JWKS download, MSAL token endpoint) are faked."""
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

import auth
from auth import AuthError, AuthPath, ServicePrincipal, authenticate_service, authenticate_user
from entra import EntraIDProvider

TENANT, CLIENT, AUD = "tenant-123", "platform-client", "api://platform-client"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
SP = ServicePrincipal(client_id="sp-rollup", service_account_id="svc-telemetry-rollup@contoso",
                      execution_role="SVC_TELEMETRY_ROLLUP_ROLE", allowed_dag_ids=frozenset({"nightly_telemetry_rollup"}))


class FakeMsalApp:
    def __init__(self, result):
        self.result, self.calls = result, []

    def acquire_token_on_behalf_of(self, user_assertion, scopes):
        self.calls.append(("obo", user_assertion, scopes))
        return self.result

    def acquire_token_for_client(self, scopes):
        self.calls.append(("client", scopes))
        return self.result


def provider(result=None, apps=None):
    result = result if result is not None else {"access_token": "snowflake-access-token", "expires_in": 3600}

    def factory(client_id, client_secret):
        app = FakeMsalApp(result)
        if apps is not None:
            apps.append((client_id, client_secret, app))
        return app

    return EntraIDProvider(TENANT, CLIENT, "s3cret", "api://snowflake-app", service_principals=[SP],
                           key_resolver=lambda token: KEY.public_key(), msal_app_factory=factory)


def token(key=KEY, alg="RS256", **overrides):
    now = int(time.time())
    claims = {"iss": f"https://login.microsoftonline.com/{TENANT}/v2.0", "aud": AUD, "tid": TENANT,
              "iat": now, "nbf": now, "exp": now + 600, "oid": "oid-9", "preferred_username": "dana.ops@contoso.com",
              "roles": ["Lakehouse.Operator"], "scp": "access_as_user"} | overrides
    return jwt.encode({k: v for k, v in claims.items() if v is not None}, key, algorithm=alg, headers={"kid": "k1"})


def test_user_path_validates_rs256_and_exchanges_on_behalf_of():
    apps = []
    ctx = authenticate_user(token(), idp=provider(apps=apps))
    assert ctx.auth_path is AuthPath.USER_OBO and ctx.principal == "dana.ops@contoso.com"
    assert ctx.snowflake_user == "DANA.OPS@CONTOSO.COM" and ctx.snowflake_role == "OPS_ANALYST"
    client_id, secret, app = apps[0]
    assert (client_id, secret) == (CLIENT, "s3cret")
    kind, assertion, scopes = app.calls[0]
    assert kind == "obo" and scopes == ["api://snowflake-app/session:role:OPS_ANALYST"] and assertion.count(".") == 2
    assert "snowflake-access-token" not in repr(ctx)  # only a fingerprint is kept


@pytest.mark.parametrize("bad, reason", [
    (dict(aud="api://someone-else"), "audience"),
    (dict(iss="https://login.microsoftonline.com/evil/v2.0"), "issuer"),
    (dict(tid="other-tenant"), "different tenant"),
    (dict(exp=int(time.time()) - 5), "expired"),
    (dict(scp="User.Read"), "access_as_user"),
    (dict(roles=["Nope"]), "no app role"),
    (dict(oid=None), "identity claims"),
])
def test_user_token_rejections(bad, reason):
    with pytest.raises(AuthError, match=reason):
        authenticate_user(token(**bad), idp=provider())


def test_signature_and_algorithm_attacks_are_rejected():
    with pytest.raises(AuthError):
        authenticate_user(token(key=OTHER_KEY), idp=provider())  # signed by another key
    with pytest.raises(AuthError):
        authenticate_user(token(key="shared-secret-shared-secret-shared-secret", alg="HS256"), idp=provider())  # algorithm downgrade
    with pytest.raises(AuthError):
        authenticate_user("not.a.jwt", idp=provider())

    def broken_resolver(tok):
        raise ConnectionError("jwks unreachable")
    idp = EntraIDProvider(TENANT, CLIENT, "s", "api://sf", key_resolver=broken_resolver, msal_app_factory=lambda a, b: None)
    with pytest.raises(AuthError, match="Signing key"):
        idp.validate_user_token(token())


def test_obo_failure_does_not_leak_error_description():
    idp = provider({"error": "invalid_grant", "error_description": "AADSTS50013: tenant internals"})
    with pytest.raises(AuthError) as err:
        authenticate_user(token(), idp=idp)
    assert "invalid_grant" in str(err.value) and "AADSTS" not in str(err.value)
    with pytest.raises(AuthError, match="no lifetime"):
        authenticate_user(token(), idp=provider({"access_token": "t", "expires_in": 0}))


def test_service_path():
    apps = []
    ctx = authenticate_service("sp-rollup", "sp-secret", "nightly_telemetry_rollup", "scheduled__2026-11-17", idp=provider(apps=apps))
    assert ctx.auth_path is AuthPath.M2M and ctx.service_account_id == "svc-telemetry-rollup@contoso"
    assert ctx.snowflake_role == "SVC_TELEMETRY_ROLLUP_ROLE" and ctx.airflow_dag_run_id == "scheduled__2026-11-17"
    assert apps[0][:2] == ("sp-rollup", "sp-secret") and apps[0][2].calls == [("client", ["api://snowflake-app/.default"])]
    with pytest.raises(AuthError):
        authenticate_service("unknown-sp", "x", "nightly_telemetry_rollup", "r1", idp=provider())
    with pytest.raises(AuthError, match="not permitted"):
        authenticate_service("sp-rollup", "sp-secret", "rogue_dag", "r1", idp=provider())
    with pytest.raises(AuthError, match="invalid_client"):
        authenticate_service("sp-rollup", "wrong", "nightly_telemetry_rollup", "r1", idp=provider({"error": "invalid_client"}))


def test_provider_selection_from_environment(monkeypatch):
    monkeypatch.setenv("IDENTITY_PROVIDER", "entra")
    for name in ("ENTRA_TENANT_ID", "ENTRA_CLIENT_ID", "ENTRA_CLIENT_SECRET", "SNOWFLAKE_SCOPE_PREFIX"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="needs tenant id"):
        auth.get_identity_provider()
    monkeypatch.setenv("ENTRA_TENANT_ID", TENANT)
    monkeypatch.setenv("ENTRA_CLIENT_ID", CLIENT)
    monkeypatch.setenv("ENTRA_CLIENT_SECRET", "s3cret")
    monkeypatch.setenv("SNOWFLAKE_SCOPE_PREFIX", "api://snowflake-app/")
    monkeypatch.setenv("SERVICE_PRINCIPALS_JSON", '[{"client_id": "sp-rollup", "service_account_id": "svc@contoso", '
                       '"execution_role": "SVC_TELEMETRY_ROLLUP_ROLE", "allowed_dag_ids": ["nightly_telemetry_rollup"]}]')
    idp = auth.get_identity_provider()
    assert idp.name == "entra" and idp.api_audience == AUD and idp.service_principal("sp-rollup").execution_role == "SVC_TELEMETRY_ROLLUP_ROLE"
    assert idp.jwks_uri == f"https://login.microsoftonline.com/{TENANT}/discovery/v2.0/keys"
    monkeypatch.setenv("IDENTITY_PROVIDER", "okta")
    with pytest.raises(RuntimeError, match="Unknown IDENTITY_PROVIDER"):
        auth.get_identity_provider()
    monkeypatch.setenv("IDENTITY_PROVIDER", "simulated")
    assert auth.get_identity_provider() is auth.IDP
