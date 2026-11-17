"""auth.py - Dual authentication paths for the platform.

User path     : Microsoft Entra ID OAuth2 On-Behalf-Of (OBO). The human's JWT
                is exchanged for a Snowflake-audience token carrying the same
                identity, so Snowflake Row-Level Security evaluates
                `WHERE user_id = CURRENT_USER` against the real person.
Headless path : Machine-to-Machine (M2M) client-credentials for scheduled
                Airflow DAGs. The service principal maps to one scoped IAM
                execution role; `service_account_id` and `airflow_dag_run_id`
                are logged on every authentication.

Identity providers (IDENTITY_PROVIDER):
  simulated (default)  `SimulatedEntraID` below: in-process issuer, HS256 tokens
                       signed with a local key. For development and evaluation.
  entra                `entra.EntraIDProvider`: RS256 validation against the
                       tenant's JWKS and token exchange through MSAL.
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field
from sqlglot import exp

from gates import SQL_DIALECT, _is_cte_ref, _scope_of, _table_fqn, parse_read_only_sql, resolve_table
from schemas import AuditEvent, Catalog, DataDomain, GateResult, GateStatus

logger = logging.getLogger("platform.auth")

TENANT_ID = "contoso-industrial-tenant"
ISSUER = f"https://login.microsoftonline.com/{TENANT_ID}/v2.0"
API_AUDIENCE = "api://enterprise-ai-platform"  # this middle-tier API
SNOWFLAKE_AUDIENCE = "https://lakehouse.snowflakecomputing.com"
OBO_REQUIRED_SCOPE = "access_as_user"
TOKEN_TTL_SECONDS = 3600
GATE_AUTHZ = "authorization"
RLS_COLUMN = "USER_ID"


class AuthError(Exception):
    """Authentication or token exchange failed. Always fail closed."""


class AuthPath(StrEnum):
    USER_OBO = "user_obo"
    M2M = "m2m"


# ---------------------------------------------------------------------------
# Role model: Entra app roles -> Snowflake roles -> data domains
# ---------------------------------------------------------------------------

APP_ROLE_TO_SNOWFLAKE_ROLE: dict[str, str] = {
    "Lakehouse.Operator": "OPS_ANALYST",
    "Lakehouse.Finance": "FIN_ANALYST",
    "Lakehouse.Admin": "PLATFORM_ADMIN",
}

SNOWFLAKE_ROLE_DOMAINS: dict[str, frozenset[DataDomain]] = {
    "OPS_ANALYST": frozenset({DataDomain.SCADA, DataDomain.MAINTENANCE}),
    "FIN_ANALYST": frozenset({DataDomain.FINANCE, DataDomain.MAINTENANCE}),
    "PLATFORM_ADMIN": frozenset(DataDomain),
    # Scoped IAM execution roles for headless DAGs
    "SVC_TELEMETRY_ROLLUP_ROLE": frozenset({DataDomain.SCADA}),
    "SVC_MAINT_REPORT_ROLE": frozenset({DataDomain.MAINTENANCE}),
}


class AuthContext(BaseModel):
    """Verified identity attached to the workflow state."""

    model_config = ConfigDict(extra="forbid")

    auth_path: AuthPath
    principal: str = Field(description="UPN for users, service_account_id for services")
    object_id: str
    snowflake_user: str
    snowflake_role: str
    allowed_domains: frozenset[DataDomain]
    scopes: tuple[str, ...] = ()
    expires_at: datetime
    token_fingerprint: str = Field(description="SHA-256 prefix of the access token")
    service_account_id: str | None = None
    airflow_dag_id: str | None = None
    airflow_dag_run_id: str | None = None

    @property
    def is_expired(self) -> bool:
        return datetime.now(UTC) >= self.expires_at

    def to_audit_event(self, node: str = "auth") -> AuditEvent:
        detail: dict[str, Any] = {
            "auth_path": self.auth_path.value,
            "principal": self.principal,
            "snowflake_user": self.snowflake_user,
            "snowflake_role": self.snowflake_role,
            "token_fingerprint": self.token_fingerprint,
        }
        if self.auth_path is AuthPath.M2M:
            detail |= {
                "service_account_id": self.service_account_id,
                "airflow_dag_id": self.airflow_dag_id,
                "airflow_dag_run_id": self.airflow_dag_run_id,
            }
        return AuditEvent(
            node=node, event_type="auth", status=GateStatus.PASSED,
            message=f"Authenticated via {self.auth_path.value}", detail=detail,
        )


# ---------------------------------------------------------------------------
# Minimal JWT (HS256) - stdlib only
# ---------------------------------------------------------------------------


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def encode_jwt(claims: dict[str, Any], key: bytes) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = f"{_b64url(json.dumps(header).encode())}.{_b64url(json.dumps(claims).encode())}"
    sig = hmac.new(key, signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(sig)}"


def decode_jwt(token: str, key: bytes, audience: str, issuer: str = ISSUER) -> dict[str, Any]:
    """Verify signature, algorithm, issuer, audience and validity window."""
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        header = json.loads(_b64url_decode(header_b64))
        claims = json.loads(_b64url_decode(payload_b64))
        signature = _b64url_decode(sig_b64)
    except (ValueError, json.JSONDecodeError) as err:
        raise AuthError("Malformed token.") from err
    if header.get("alg") != "HS256":  # blocks alg=none and algorithm confusion
        raise AuthError(f"Unsupported signing algorithm: {header.get('alg')!r}.")
    expected = hmac.new(key, f"{header_b64}.{payload_b64}".encode(), hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise AuthError("Token signature is invalid.")
    now = time.time()
    if not isinstance(claims.get("exp"), (int, float)) or now >= claims["exp"]:
        raise AuthError("Token is expired.")
    if now < claims.get("nbf", 0):
        raise AuthError("Token is not yet valid.")
    if claims.get("iss") != issuer:
        raise AuthError("Token issuer is not trusted.")
    if claims.get("aud") != audience:
        raise AuthError(f"Token audience mismatch (expected {audience}).")
    return claims


def _fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Simulated Entra ID
# ---------------------------------------------------------------------------


class ServicePrincipal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    client_id: str
    service_account_id: str
    execution_role: str
    allowed_dag_ids: frozenset[str]
    secret_sha256: str | None = Field(default=None, description="Simulated provider only")


class DownstreamToken(BaseModel):
    """Access token for Snowflake obtained by OBO or client credentials."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    access_token: str = Field(repr=False)
    expires_at: datetime
    scopes: tuple[str, ...] = ()


class TokenVault:
    """In-process store of downstream access tokens, keyed by fingerprint.

    Workflow state and checkpoints carry only the fingerprint; the warehouse
    executor exchanges it here for the real token. A token that is missing
    (another replica, a restart) or expired raises AuthError, so the query is
    refused rather than run under a different identity."""

    def __init__(self) -> None:
        self._tokens: dict[str, DownstreamToken] = {}
        self._owners: dict[str, str] = {}
        self._lock = threading.Lock()

    def put(self, token: DownstreamToken, principal: str) -> str:
        fingerprint = _fingerprint(token.access_token)
        now = datetime.now(UTC)
        with self._lock:
            for key in [k for k, t in self._tokens.items() if t.expires_at <= now]:
                del self._tokens[key]
                self._owners.pop(key, None)
            self._tokens[fingerprint] = token
            self._owners[fingerprint] = principal
        return fingerprint

    def get(self, fingerprint: str, principal: str) -> DownstreamToken:
        """The token for this fingerprint, only for the principal it was issued to."""
        with self._lock:
            token = self._tokens.get(fingerprint)
            if token is not None and token.expires_at <= datetime.now(UTC):
                del self._tokens[fingerprint]
                self._owners.pop(fingerprint, None)
                token = None
            if token is not None and not hmac.compare_digest(self._owners.get(fingerprint, ""), principal):
                token = None
        if token is None:
            raise AuthError("No valid warehouse token for this session; sign in again.")
        return token

    def __len__(self) -> int:
        with self._lock:
            return len(self._tokens)


TOKEN_VAULT = TokenVault()


class IdentityProvider(Protocol):
    """What the platform needs from an identity provider. Implemented by
    `SimulatedEntraID` here and by `entra.EntraIDProvider` for production."""

    name: str

    def validate_user_token(self, token: str) -> dict[str, Any]: ...

    def exchange_on_behalf_of(self, user_token: str, scope: str) -> DownstreamToken: ...

    def acquire_service_token(self, client_id: str, client_secret: str) -> DownstreamToken: ...

    def service_principal(self, client_id: str) -> ServicePrincipal | None: ...


class SimulatedEntraID:
    """In-process stand-in for the Entra ID token endpoint."""

    name = "simulated"

    def __init__(self, signing_key: bytes | None = None) -> None:
        self._key = signing_key or secrets.token_bytes(32)
        self._platform_client: tuple[str, str] | None = None
        self._confidential_clients: dict[str, str] = {}  # client_id -> secret hash
        self._service_principals: dict[str, ServicePrincipal] = {}

    @staticmethod
    def _hash(secret: str) -> str:
        return hashlib.sha256(secret.encode()).hexdigest()

    def register_confidential_client(self, client_id: str, client_secret: str) -> None:
        """Register this API's own app registration (the OBO middle tier)."""
        self._confidential_clients[client_id] = self._hash(client_secret)
        self._platform_client = (client_id, client_secret)

    def register_service_principal(
        self, client_id: str, client_secret: str, service_account_id: str,
        execution_role: str, allowed_dag_ids: set[str],
    ) -> None:
        if execution_role not in SNOWFLAKE_ROLE_DOMAINS:
            raise ValueError(f"Unknown execution role: {execution_role}")
        self._service_principals[client_id] = ServicePrincipal(
            client_id=client_id, secret_sha256=self._hash(client_secret),
            service_account_id=service_account_id, execution_role=execution_role,
            allowed_dag_ids=frozenset(allowed_dag_ids),
        )

    def _base_claims(self, audience: str, ttl: int = TOKEN_TTL_SECONDS) -> dict[str, Any]:
        now = int(time.time())
        return {
            "iss": ISSUER, "aud": audience, "tid": TENANT_ID,
            "iat": now, "nbf": now, "exp": now + ttl, "jti": secrets.token_hex(8),
        }

    def issue_user_token(
        self, upn: str, oid: str, roles: list[str], ttl: int = TOKEN_TTL_SECONDS
    ) -> str:
        """Simulates interactive sign-in: a user token for the middle-tier API."""
        claims = self._base_claims(API_AUDIENCE, ttl) | {
            "sub": oid, "oid": oid, "upn": upn, "roles": roles, "scp": OBO_REQUIRED_SCOPE,
        }
        return encode_jwt(claims, self._key)

    def acquire_token_on_behalf_of(
        self, user_assertion: str, client_id: str, client_secret: str, scope: str
    ) -> str:
        """OAuth2 OBO grant (urn:ietf:params:oauth:grant-type:jwt-bearer).

        Validates the middle tier's own credentials and the inbound user
        assertion, then issues a downstream token with the user's identity.
        """
        stored = self._confidential_clients.get(client_id)
        if stored is None or not hmac.compare_digest(stored, self._hash(client_secret)):
            raise AuthError("Confidential client authentication failed.")
        user = decode_jwt(user_assertion, self._key, audience=API_AUDIENCE)
        if OBO_REQUIRED_SCOPE not in str(user.get("scp", "")).split():
            raise AuthError(f"User token lacks the '{OBO_REQUIRED_SCOPE}' scope.")
        if not user.get("upn") or not user.get("oid"):
            raise AuthError("User token is missing identity claims (upn/oid).")
        remaining = int(user["exp"] - time.time())
        claims = self._base_claims(SNOWFLAKE_AUDIENCE, min(TOKEN_TTL_SECONDS, remaining)) | {
            "sub": user["oid"], "oid": user["oid"], "upn": user["upn"],
            "roles": user.get("roles", []), "scp": scope, "azp": client_id,
        }
        return encode_jwt(claims, self._key)

    def acquire_token_for_client(self, client_id: str, client_secret: str) -> str:
        """OAuth2 client-credentials grant for a service principal (app-only)."""
        sp = self._service_principals.get(client_id)
        if sp is None or not hmac.compare_digest(sp.secret_sha256, self._hash(client_secret)):
            raise AuthError("Service principal authentication failed.")
        claims = self._base_claims(SNOWFLAKE_AUDIENCE) | {
            "sub": sp.service_account_id, "oid": sp.service_account_id,
            "azp": client_id, "idtyp": "app", "roles": [sp.execution_role],
        }
        return encode_jwt(claims, self._key)

    def validate(self, token: str, audience: str) -> dict[str, Any]:
        return decode_jwt(token, self._key, audience=audience)

    def service_principal(self, client_id: str) -> ServicePrincipal | None:
        return self._service_principals.get(client_id)

    # -- IdentityProvider interface -----------------------------------------

    def _downstream(self, token: str) -> DownstreamToken:
        claims = self.validate(token, SNOWFLAKE_AUDIENCE)
        return DownstreamToken(
            access_token=token,
            expires_at=datetime.fromtimestamp(claims["exp"], UTC),
            scopes=tuple(str(claims.get("scp", "")).split()),
        )

    def validate_user_token(self, token: str) -> dict[str, Any]:
        return self.validate(token, API_AUDIENCE)

    def exchange_on_behalf_of(self, user_token: str, scope: str) -> DownstreamToken:
        if self._platform_client is None:
            raise AuthError("No confidential client is registered for the OBO exchange.")
        client_id, client_secret = self._platform_client
        return self._downstream(
            self.acquire_token_on_behalf_of(user_token, client_id, client_secret, scope)
        )

    def acquire_service_token(self, client_id: str, client_secret: str) -> DownstreamToken:
        return self._downstream(self.acquire_token_for_client(client_id, client_secret))


# ---------------------------------------------------------------------------
# Demo tenant wiring (override secrets through the environment)
# ---------------------------------------------------------------------------

PLATFORM_CLIENT_ID = "enterprise-ai-platform-api"
PLATFORM_CLIENT_SECRET = os.environ.get("PLATFORM_CLIENT_SECRET", "dev-only-platform-secret")
TELEMETRY_SP_CLIENT_ID = "sp-telemetry-rollup"
TELEMETRY_SP_SECRET = os.environ.get("TELEMETRY_SP_SECRET", "dev-only-telemetry-secret")

IDP = SimulatedEntraID()
IDP.register_confidential_client(PLATFORM_CLIENT_ID, PLATFORM_CLIENT_SECRET)
IDP.register_service_principal(
    client_id=TELEMETRY_SP_CLIENT_ID,
    client_secret=TELEMETRY_SP_SECRET,
    service_account_id="svc-telemetry-rollup@contoso-industrial",
    execution_role="SVC_TELEMETRY_ROLLUP_ROLE",
    allowed_dag_ids={"nightly_telemetry_rollup", "hourly_anomaly_scan"},
)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _snowflake_role_for(app_roles: list[str]) -> str:
    for app_role in app_roles:
        if role := APP_ROLE_TO_SNOWFLAKE_ROLE.get(app_role):
            return role
    raise AuthError("User holds no app role that maps to a Snowflake role.")


def get_identity_provider() -> IdentityProvider:
    """Provider named by IDENTITY_PROVIDER: `simulated` (default) or `entra`."""
    choice = os.environ.get("IDENTITY_PROVIDER", "simulated").lower()
    if choice == "simulated":
        return IDP
    if choice == "entra":
        from entra import EntraIDProvider  # imported lazily: needs msal and PyJWT

        return EntraIDProvider.from_env()
    raise RuntimeError(f"Unknown IDENTITY_PROVIDER '{choice}' (use simulated or entra).")


ACTIVE_IDP: IdentityProvider = get_identity_provider()


def authenticate_user(user_jwt: str, idp: IdentityProvider | None = None) -> AuthContext:
    """User path: validate the human's JWT, run the OBO exchange, and build a
    Snowflake session identity that carries the human's claims."""
    idp = idp or ACTIVE_IDP
    try:
        claims = idp.validate_user_token(user_jwt)
        upn = claims.get("upn")  # preferred_username is mutable and not safe for authorization
        oid = claims.get("oid")
        if not upn or not oid:
            raise AuthError("User token is missing identity claims (upn/oid).")
        if OBO_REQUIRED_SCOPE not in str(claims.get("scp", "")).split():
            raise AuthError(f"User token lacks the '{OBO_REQUIRED_SCOPE}' scope.")
        role = _snowflake_role_for(claims.get("roles", []))
        token = idp.exchange_on_behalf_of(user_jwt, f"session:role:{role}")
    except AuthError as err:
        logger.warning("auth.user_obo.denied provider=%s reason=%s", idp.name, err)
        raise
    user_expiry = datetime.fromtimestamp(claims["exp"], UTC)
    ctx = AuthContext(
        auth_path=AuthPath.USER_OBO,
        principal=upn,
        object_id=oid,
        snowflake_user=upn.upper(),  # Snowflake LOGIN_NAME = UPN
        snowflake_role=role,
        allowed_domains=SNOWFLAKE_ROLE_DOMAINS[role],
        scopes=token.scopes,
        expires_at=min(token.expires_at, user_expiry),
        token_fingerprint=TOKEN_VAULT.put(token, upn),
    )
    logger.info(
        "auth.user_obo.ok provider=%s principal=%s role=%s", idp.name, ctx.principal, ctx.snowflake_role
    )
    return ctx


def authenticate_service(
    client_id: str, client_secret: str, airflow_dag_id: str, airflow_dag_run_id: str,
    idp: IdentityProvider | None = None,
) -> AuthContext:
    """Headless path: client-credentials for a scheduled DAG, bound to one
    scoped execution role and to the DAGs the principal is registered for."""
    idp = idp or ACTIVE_IDP
    try:
        token = idp.acquire_service_token(client_id, client_secret)
        sp = idp.service_principal(client_id)
        if sp is None:
            raise AuthError("Service principal is not registered with the platform.")
        if airflow_dag_id not in sp.allowed_dag_ids:
            raise AuthError(f"DAG '{airflow_dag_id}' is not permitted for this service principal.")
        if not airflow_dag_run_id.strip():
            raise AuthError("airflow_dag_run_id is required for headless execution.")
    except AuthError as err:
        logger.warning(
            "auth.m2m.denied provider=%s client_id=%s airflow_dag_id=%s airflow_dag_run_id=%s reason=%s",
            idp.name, client_id, airflow_dag_id, airflow_dag_run_id, err,
        )
        raise
    ctx = AuthContext(
        auth_path=AuthPath.M2M,
        principal=sp.service_account_id,
        object_id=sp.service_account_id,
        snowflake_user=sp.service_account_id.split("@")[0].upper().replace("-", "_"),
        snowflake_role=sp.execution_role,
        allowed_domains=SNOWFLAKE_ROLE_DOMAINS[sp.execution_role],
        scopes=token.scopes,
        expires_at=token.expires_at,
        token_fingerprint=TOKEN_VAULT.put(token, sp.service_account_id),
        service_account_id=sp.service_account_id,
        airflow_dag_id=airflow_dag_id,
        airflow_dag_run_id=airflow_dag_run_id,
    )
    logger.info(
        "auth.m2m.ok provider=%s service_account_id=%s airflow_dag_run_id=%s role=%s",
        idp.name, ctx.service_account_id, ctx.airflow_dag_run_id, ctx.snowflake_role,
    )
    return ctx


def load_auth_context(raw: dict[str, Any] | AuthContext | None) -> AuthContext:
    """Re-validate the context stored in workflow state. Missing, malformed or
    expired contexts raise AuthError."""
    if raw is None or raw == {}:
        raise AuthError("No authenticated identity on the request.")
    try:
        ctx = raw if isinstance(raw, AuthContext) else AuthContext.model_validate(raw)
    except ValueError as err:
        raise AuthError("Auth context is malformed.") from err
    if ctx.is_expired:
        raise AuthError("Auth context has expired.")
    if ctx.auth_path is AuthPath.M2M and not (ctx.service_account_id and ctx.airflow_dag_run_id):
        raise AuthError("Headless context lacks service_account_id/airflow_dag_run_id.")
    return ctx


# ---------------------------------------------------------------------------
# Authorization and Row-Level Security
# ---------------------------------------------------------------------------


def authorize_sql(sql: str, ctx: AuthContext, catalog: Catalog) -> GateResult:
    """The session role must be entitled to every table's data domain."""
    root = parse_read_only_sql(sql)
    denied: list[str] = []
    for table in root.find_all(exp.Table):
        if _is_cte_ref(table):
            continue
        spec = resolve_table(table, catalog)
        if spec is None:
            denied.append(f"Table '{_table_fqn(table)}' is not in the catalog.")
        elif spec.domain not in ctx.allowed_domains:
            denied.append(
                f"Role {ctx.snowflake_role} is not entitled to {spec.domain.value} table '{spec.fqn}'."
            )
    return GateResult(
        gate=GATE_AUTHZ,
        status=GateStatus.REFUSED if denied else GateStatus.PASSED,
        reasons=list(dict.fromkeys(denied)),
        details={"snowflake_role": ctx.snowflake_role, "auth_path": ctx.auth_path.value},
    )


def apply_row_level_security(sql: str, ctx: AuthContext, catalog: Catalog) -> str:
    """Add `<alias>.USER_ID = CURRENT_USER()` to every SELECT that reads a
    table carrying USER_ID, on the user path.

    In production this predicate lives in a Snowflake ROW ACCESS POLICY and
    is enforced by the warehouse because the session is opened with the OBO
    token. The rewrite here makes the same guarantee visible and testable in
    the simulation. Service principals are constrained by their scoped
    execution role instead, so no predicate is added for them.

    On both paths the returned SQL is generated from the parsed, validated
    AST, so what executes is exactly what the gates inspected.
    """
    root = parse_read_only_sql(sql)
    if ctx.auth_path is AuthPath.M2M:
        # No predicate is added, but the warehouse still receives the SQL as re-serialised
        # from the validated AST, never the model's raw string.
        return root.sql(dialect=SQL_DIALECT, comments=False)
    for select in root.find_all(exp.Select):
        for alias, spec in _scope_of(select, catalog).items():
            if RLS_COLUMN in spec.columns:
                select.where(
                    exp.column(RLS_COLUMN, table=alias).eq(exp.CurrentUser()), copy=False
                )
    return root.sql(dialect=SQL_DIALECT, comments=False)
