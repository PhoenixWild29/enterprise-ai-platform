"""entra.py - Microsoft Entra ID identity provider (RS256 + MSAL).

Production counterpart of `auth.SimulatedEntraID`. It implements the same
four operations the auth module needs:

  validate_user_token     RS256 signature via the tenant's JWKS, issuer,
                          audience, tenant and expiry
  exchange_on_behalf_of   OAuth2 OBO grant through MSAL
  acquire_service_token   client-credentials grant through MSAL
  service_principal       local registry: which DAGs and which scoped
                          execution role each service principal may use

Configuration (EntraIDProvider.from_env):
  ENTRA_TENANT_ID, ENTRA_CLIENT_ID, ENTRA_CLIENT_SECRET   this API's app registration
  ENTRA_API_AUDIENCE          audience of inbound user tokens (default api://<client id>)
  ENTRA_ALLOWED_CLIENT_IDS    optional comma-separated allow-list for the token's azp/appid

The app registration must issue v2.0 tokens with the optional `upn` claim;
`preferred_username` is mutable and is not accepted as an identity.
  SNOWFLAKE_SCOPE_PREFIX      Snowflake OAuth resource, e.g. api://<snowflake app id>
  SERVICE_PRINCIPALS_JSON     [{"client_id": "...", "service_account_id": "...",
                                "execution_role": "...", "allowed_dag_ids": ["..."]}]
"""

import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from auth import SNOWFLAKE_ROLE_DOMAINS, AuthError, DownstreamToken, ServicePrincipal

type KeyResolver = Callable[[str], Any]  # token -> public key
type MsalAppFactory = Callable[[str, str], Any]  # (client_id, client_secret) -> ConfidentialClientApplication

AUTHORITY_HOST = "https://login.microsoftonline.com"
CLOCK_SKEW_SECONDS = 60


class EntraIDProvider:
    name = "entra"

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        snowflake_scope_prefix: str,
        api_audience: str | None = None,
        service_principals: list[ServicePrincipal] | None = None,
        allowed_client_ids: list[str] | None = None,
        *,
        key_resolver: KeyResolver | None = None,
        msal_app_factory: MsalAppFactory | None = None,
    ) -> None:
        if not all((tenant_id, client_id, client_secret, snowflake_scope_prefix)):
            raise RuntimeError("Entra provider needs tenant id, client id, client secret and Snowflake scope prefix.")
        self.tenant_id = tenant_id
        self.client_id = client_id
        self._client_secret = client_secret
        self.api_audience = api_audience or f"api://{client_id}"
        self.snowflake_scope_prefix = snowflake_scope_prefix.rstrip("/")
        self.authority = f"{AUTHORITY_HOST}/{tenant_id}"
        self.issuer = f"{self.authority}/v2.0"
        self.jwks_uri = f"{self.authority}/discovery/v2.0/keys"
        self._key_resolver = key_resolver or self._default_key_resolver()
        self._msal_app_factory = msal_app_factory or self._default_msal_app
        self._platform_app: Any = None
        self.allowed_client_ids = frozenset(allowed_client_ids or ())
        self._service_principals: dict[str, ServicePrincipal] = {}
        for sp in service_principals or []:
            if sp.execution_role not in SNOWFLAKE_ROLE_DOMAINS:
                raise RuntimeError(f"Unknown execution role '{sp.execution_role}' for {sp.client_id}.")
            self._service_principals[sp.client_id] = sp

    # -- construction -------------------------------------------------------

    @classmethod
    def from_env(cls) -> "EntraIDProvider":
        try:
            raw = json.loads(os.environ.get("SERVICE_PRINCIPALS_JSON", "[]"))
            principals = [ServicePrincipal.model_validate(item) for item in raw]
        except ValueError as err:
            raise RuntimeError(f"SERVICE_PRINCIPALS_JSON is invalid: {err}") from err
        return cls(
            tenant_id=os.environ.get("ENTRA_TENANT_ID", ""),
            client_id=os.environ.get("ENTRA_CLIENT_ID", ""),
            client_secret=os.environ.get("ENTRA_CLIENT_SECRET", ""),
            snowflake_scope_prefix=os.environ.get("SNOWFLAKE_SCOPE_PREFIX", ""),
            api_audience=os.environ.get("ENTRA_API_AUDIENCE") or None,
            service_principals=principals,
            allowed_client_ids=[c for c in os.environ.get("ENTRA_ALLOWED_CLIENT_IDS", "").split(",") if c.strip()],
        )

    def _default_key_resolver(self) -> KeyResolver:
        client: Any = None

        def resolve(token: str) -> Any:
            nonlocal client
            if client is None:
                from jwt import PyJWKClient

                client = PyJWKClient(self.jwks_uri, cache_keys=True, lifespan=3600)
            return client.get_signing_key_from_jwt(token).key

        return resolve

    def _default_msal_app(self, client_id: str, client_secret: str) -> Any:
        import msal

        return msal.ConfidentialClientApplication(
            client_id, client_credential=client_secret, authority=self.authority
        )

    def _platform(self) -> Any:
        # Only this API's own app registration is cached; its secret is fixed for the process.
        if self._platform_app is None:
            self._platform_app = self._msal_app_factory(self.client_id, self._client_secret)
        return self._platform_app

    # -- identity provider operations ---------------------------------------

    def validate_user_token(self, token: str) -> dict[str, Any]:
        import jwt

        try:
            key = self._key_resolver(token)
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256"],  # never accept HS256/none for Entra tokens
                audience=self.api_audience,
                issuer=self.issuer,
                options={"require": ["exp", "iat", "iss", "aud"]},
                leeway=CLOCK_SKEW_SECONDS,
            )
        except jwt.ExpiredSignatureError as err:
            raise AuthError("Token is expired.") from err
        except jwt.InvalidAudienceError as err:
            raise AuthError(f"Token audience mismatch (expected {self.api_audience}).") from err
        except jwt.InvalidIssuerError as err:
            raise AuthError("Token issuer is not trusted.") from err
        except jwt.PyJWTError as err:
            raise AuthError(f"Token validation failed ({type(err).__name__}).") from err
        except Exception as err:  # noqa: BLE001 - JWKS fetch or key errors must not pass a token
            raise AuthError("Signing key could not be resolved.") from err
        if claims.get("tid") != self.tenant_id:
            raise AuthError("Token was issued for a different tenant.")
        if self.allowed_client_ids and (claims.get("azp") or claims.get("appid")) not in self.allowed_client_ids:
            raise AuthError("Token was issued to a client application that is not allowed.")
        return claims

    @staticmethod
    def _to_token(result: dict[str, Any], scopes: tuple[str, ...], action: str) -> DownstreamToken:
        if not isinstance(result, dict) or "access_token" not in result:
            code = result.get("error", "unknown_error") if isinstance(result, dict) else "unknown_error"
            raise AuthError(f"{action} failed ({code}).")  # error_description can carry tenant detail; keep it out
        expires_in = int(result.get("expires_in", 0))
        if expires_in <= 0:
            raise AuthError(f"{action} returned a token with no lifetime.")
        return DownstreamToken(
            access_token=result["access_token"],
            expires_at=datetime.fromtimestamp(time.time() + expires_in, UTC),
            scopes=scopes,
        )

    def exchange_on_behalf_of(self, user_token: str, scope: str) -> DownstreamToken:
        target = f"{self.snowflake_scope_prefix}/{scope}"
        result = self._platform().acquire_token_on_behalf_of(user_assertion=user_token, scopes=[target])
        return self._to_token(result, (scope,), "On-behalf-of exchange")

    def acquire_service_token(self, client_id: str, client_secret: str) -> DownstreamToken:
        if client_id not in self._service_principals:
            raise AuthError("Service principal authentication failed.")
        target = f"{self.snowflake_scope_prefix}/.default"
        # A fresh MSAL app per call: nothing is cached per presented secret, so a wrong
        # or revoked secret is always checked by Entra and failed attempts leave no state.
        app = self._msal_app_factory(client_id, client_secret)
        result = app.acquire_token_for_client(scopes=[target])
        return self._to_token(result, (".default",), "Service principal authentication")

    def service_principal(self, client_id: str) -> ServicePrincipal | None:
        return self._service_principals.get(client_id)
