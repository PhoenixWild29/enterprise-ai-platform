"""agentcore_identity.py - Identity through Amazon Bedrock AgentCore Identity.

Inside AgentCore Runtime the inbound user JWT is checked by the runtime's
authorizer and turned into a workload access token. This provider then asks
AgentCore Identity for the downstream Snowflake token:

  user path      auth_flow ON_BEHALF_OF_TOKEN_EXCHANGE: the token carries the
                 person's identity, so Snowflake row-level security applies
  headless path  auth_flow M2M: the job's own credential provider, mapped to
                 one scoped execution role and its registered DAGs

The inbound token is still validated here (RS256, issuer, audience, tenant)
by the same code the Entra provider uses: defence in depth, and the claims
are needed to map roles.

Configuration:
  AGENTCORE_OBO_PROVIDER      credential provider name for the user path
  AGENTCORE_M2M_PROVIDER      credential provider name for scheduled jobs
  SNOWFLAKE_SCOPE_PREFIX      Snowflake OAuth resource, e.g. api://<snowflake app id>
  plus the ENTRA_* and SERVICE_PRINCIPALS_JSON settings described in entra.py
"""

import asyncio
import base64
import concurrent.futures
import contextvars
import json
import os
import time
from datetime import UTC, datetime
from typing import Any

from auth import AuthError, DownstreamToken, ServicePrincipal

DEFAULT_TOKEN_LIFETIME_SECONDS = 900  # used only when the token does not state its own expiry


def _unverified_expiry(token: str) -> datetime:
    """Expiry for the token vault. The token is opaque to us (Snowflake
    verifies it), so its `exp` is read without verification and only ever
    used to shorten how long we keep it."""
    ceiling = time.time() + DEFAULT_TOKEN_LIFETIME_SECONDS
    try:
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        exp = float(claims["exp"])
        if exp > time.time():
            ceiling = min(exp, time.time() + 3600)
    except (IndexError, KeyError, ValueError, TypeError):
        pass
    return datetime.fromtimestamp(ceiling, UTC)


def _run(coro: Any) -> Any:
    """Run an AgentCore SDK coroutine from sync code, with or without a running loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    ctx = contextvars.copy_context()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(ctx.run, asyncio.run, coro).result()


class AgentCoreIdentityProvider:
    name = "agentcore"

    def __init__(
        self,
        validator: Any,
        obo_provider: str,
        m2m_provider: str,
        snowflake_scope_prefix: str,
        service_principals: list[ServicePrincipal] | None = None,
        *,
        identity_client: Any = None,
        workload_token: Any = None,
        region: str | None = None,
    ) -> None:
        if not all((obo_provider, m2m_provider, snowflake_scope_prefix)):
            raise RuntimeError("AgentCore identity needs AGENTCORE_OBO_PROVIDER, AGENTCORE_M2M_PROVIDER and SNOWFLAKE_SCOPE_PREFIX.")
        self._validator = validator
        self._obo_provider, self._m2m_provider = obo_provider, m2m_provider
        self._scope_prefix = snowflake_scope_prefix.rstrip("/")
        self._service_principals = {sp.client_id: sp for sp in service_principals or []}
        self._client = identity_client
        self._workload_token = workload_token or self._context_workload_token
        self._region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")

    @classmethod
    def from_env(cls) -> "AgentCoreIdentityProvider":
        from entra import EntraIDProvider

        validator = EntraIDProvider.from_env()
        return cls(
            validator=validator,
            obo_provider=os.environ.get("AGENTCORE_OBO_PROVIDER", ""),
            m2m_provider=os.environ.get("AGENTCORE_M2M_PROVIDER", ""),
            snowflake_scope_prefix=os.environ.get("SNOWFLAKE_SCOPE_PREFIX", ""),
            service_principals=list(validator._service_principals.values()),
        )

    @staticmethod
    def _context_workload_token() -> str | None:
        from bedrock_agentcore.runtime import BedrockAgentCoreContext

        return BedrockAgentCoreContext.get_workload_access_token()

    def _identity_client(self) -> Any:
        if self._client is None:
            from bedrock_agentcore.services.identity import IdentityClient

            self._client = IdentityClient(self._region)
        return self._client

    def _token(self, provider: str, scope: str, flow: str, action: str) -> DownstreamToken:
        workload_token = self._workload_token()
        if not workload_token:
            raise AuthError(f"{action} failed: no AgentCore workload access token on this request.")
        try:
            access_token = _run(
                self._identity_client().get_token(
                    provider_name=provider, scopes=[f"{self._scope_prefix}/{scope}"],
                    agent_identity_token=workload_token, auth_flow=flow,
                )
            )
        except Exception as err:  # noqa: BLE001 - provider messages can carry tenant detail; keep them out
            raise AuthError(f"{action} failed ({type(err).__name__}).") from err
        if not isinstance(access_token, str) or not access_token:
            raise AuthError(f"{action} returned no token.")
        return DownstreamToken(access_token=access_token, expires_at=_unverified_expiry(access_token), scopes=(scope,))

    # -- IdentityProvider interface -----------------------------------------

    def validate_user_token(self, token: str) -> dict[str, Any]:
        return self._validator.validate_user_token(token)

    def exchange_on_behalf_of(self, user_token: str, scope: str) -> DownstreamToken:
        return self._token(self._obo_provider, scope, "ON_BEHALF_OF_TOKEN_EXCHANGE", "On-behalf-of exchange")

    def acquire_service_token(self, client_id: str, client_secret: str) -> DownstreamToken:
        # The job's secret lives in the AgentCore credential provider, not in the request,
        # so `client_secret` is not used here; the caller is identified by its workload token.
        if client_id not in self._service_principals:
            raise AuthError("Service principal authentication failed.")
        return self._token(self._m2m_provider, ".default", "M2M", "Service principal authentication")

    def service_principal(self, client_id: str) -> ServicePrincipal | None:
        return self._service_principals.get(client_id)
