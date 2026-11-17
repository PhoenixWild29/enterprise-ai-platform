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
              "iat": now, "nbf": now, "exp": now + 600, "oid": "oid-9", "upn": "dana.ops@contoso.com",
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
    (dict(exp=int(time.time()) - 300), "expired"),
    (dict(upn=None, preferred_username="dana.ops@contoso.com"), "identity claims"),
    (dict(scp="User.Read"), "access_as_user"),
    (dict(roles=["Nope"]), "no app role"),
    (dict(oid=None), "identity claims"),
])
def test_user_token_rejections(bad, reason):
    with pytest.raises(AuthError, match=reason):
        authenticate_user(token(**bad), idp=provider())


def _forged(alg, sign):
    """A token with valid claims, re-signed by an attacker under another algorithm."""
    import base64, json

    def b64(raw):
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    _, payload, _ = token().split(".")
    signing_input = f"{b64(json.dumps({'alg': alg, 'typ': 'JWT', 'kid': 'k1'}).encode())}.{payload}"
    return f"{signing_input}.{b64(sign(signing_input.encode()))}"


def test_signature_and_algorithm_attacks_are_rejected():
    import hashlib, hmac
    from cryptography.hazmat.primitives import serialization

    with pytest.raises(AuthError, match="InvalidSignatureError"):
        authenticate_user(token(key=OTHER_KEY), idp=provider())  # signed by another key
    # Algorithm confusion: HS256 keyed with the provider's own public key.
    public_pem = KEY.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    confused = _forged("HS256", lambda data: hmac.new(public_pem, data, hashlib.sha256).digest())
    with pytest.raises(AuthError, match="InvalidAlgorithmError"):
        authenticate_user(confused, idp=provider())
    with pytest.raises(AuthError, match="InvalidAlgorithmError"):
        authenticate_user(_forged("none", lambda data: b""), idp=provider())
    with pytest.raises(AuthError):
        authenticate_user("not.a.jwt", idp=provider())
    assert provider().validate_user_token(token(exp=int(time.time()) - 20))["oid"] == "oid-9"  # inside clock-skew leeway

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


def test_service_secrets_are_never_cached():
    apps = []
    idp = provider(apps=apps)
    for secret in ("right-secret", "right-secret", "wrong-1", "wrong-2"):
        authenticate_service("sp-rollup", secret, "nightly_telemetry_rollup", "r1", idp=idp)
    # Every call reaches the identity provider with the presented secret, and nothing is kept per secret.
    assert [a[1] for a in apps] == ["right-secret", "right-secret", "wrong-1", "wrong-2"]
    assert not hasattr(idp, "_apps") and idp._platform_app is None
    authenticate_user(token(), idp=idp)
    authenticate_user(token(), idp=idp)
    assert [a[:2] for a in apps[4:]] == [(CLIENT, "s3cret")]  # the platform's own app is built once


def test_client_application_allow_list():
    idp = EntraIDProvider(TENANT, CLIENT, "s3cret", "api://snowflake-app", allowed_client_ids=["spa-frontend"],
                          key_resolver=lambda t: KEY.public_key(),
                          msal_app_factory=lambda a, b: FakeMsalApp({"access_token": "x", "expires_in": 60}))
    assert authenticate_user(token(azp="spa-frontend"), idp=idp).principal == "dana.ops@contoso.com"
    for bad in (dict(azp="some-other-app"), dict()):
        with pytest.raises(AuthError, match="client application"):
            authenticate_user(token(**bad), idp=idp)


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
