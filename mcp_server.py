"""mcp_server.py - The platform's data tools as a Model Context Protocol server.

Any MCP client, including an agent behind Amazon Bedrock AgentCore Gateway,
can call these tools. Each call is authenticated with the caller's bearer
token and runs the same gates as every other entry point:

  find_tables      governed tables the caller may query for a question
  search_manuals   passages from manuals and procedures (thin-set gated)
  run_sql          one read-only SELECT, run as the caller (all SQL gates,
                   authorization, row-level security)

These tools return evidence, not answers. A client that writes prose from
the evidence is responsible for its own grounding; the Strands agent and
the LangGraph workflow in this repository add the citation gate for that.

Run:  python mcp_server.py      (streamable HTTP; set MCP_RESOURCE_URL and MCP_ISSUER_URL)
"""

import hashlib
import os
import secrets
import threading
import time

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer

from auth import ACTIVE_IDP, AuthContext, AuthError, IdentityProvider, authenticate_user
from graph import AgentDeps
from platform_tools import PlatformTools, RunLedger

CONTEXT_CACHE_SIZE = 512


class PlatformTokenVerifier:
    """Validates the caller's JWT and runs the on-behalf-of exchange once per
    token. The resulting identity is kept only until the token expires."""

    def __init__(self, idp: IdentityProvider | None = None, resource_url: str | None = None) -> None:
        self._idp = idp or ACTIVE_IDP
        self._resource_url = resource_url
        self._contexts: dict[str, AuthContext] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    async def verify_token(self, token: str) -> AccessToken | None:
        ctx = self.context_for(token)
        if ctx is None:
            return None
        return AccessToken(
            token=token, client_id=ctx.object_id, scopes=list(ctx.scopes), expires_at=int(ctx.expires_at.timestamp()),
            resource=self._resource_url, subject=ctx.principal,
        )

    def context_for(self, token: str) -> AuthContext | None:
        key = self._key(token)
        with self._lock:
            ctx = self._contexts.get(key)
            if ctx is not None and ctx.is_expired:
                del self._contexts[key]
                ctx = None
        if ctx is None:
            try:
                ctx = authenticate_user(token, idp=self._idp)
            except AuthError:
                return None
            with self._lock:
                if len(self._contexts) >= CONTEXT_CACHE_SIZE:
                    now = time.time()
                    for stale in [k for k, c in self._contexts.items() if c.expires_at.timestamp() <= now]:
                        del self._contexts[stale]
                    if len(self._contexts) >= CONTEXT_CACHE_SIZE:
                        self._contexts.clear()
                self._contexts[key] = ctx
        return ctx


def build_server(deps: AgentDeps | None = None, verifier: PlatformTokenVerifier | None = None) -> MCPServer:
    deps = deps or AgentDeps()
    resource_url = os.environ.get("MCP_RESOURCE_URL", "http://127.0.0.1:8000/mcp")
    verifier = verifier or PlatformTokenVerifier(resource_url=resource_url)
    server = MCPServer(
        name="enterprise-ai-platform",
        instructions="Governed, read-only access to the lakehouse and to operating procedures. Every call runs as the signed-in user.",
        token_verifier=verifier,
        auth=AuthSettings(
            issuer_url=os.environ.get("MCP_ISSUER_URL", "http://127.0.0.1:8000"),
            resource_server_url=resource_url,
            # The verifier checks audience, issuer and tenant itself, through the identity provider.
            validate_token_resource=False,
        ),
    )

    def tools_for_caller(question: str) -> PlatformTools:
        """Fail closed: no verified identity, no tool."""
        access = get_access_token()
        ctx = verifier.context_for(access.token) if access else None
        if ctx is None:
            raise PermissionError("Not authenticated.")
        return PlatformTools(deps, RunLedger(query=question, ctx=ctx, trace_id=secrets.token_hex(16)))

    @server.tool()
    def find_tables(question: str) -> str:
        """List the governed tables relevant to a question that the caller may query, with keys and time columns."""
        return tools_for_caller(question).find_tables(question)

    @server.tool()
    def search_manuals(query: str) -> str:
        """Search manuals, procedures and policies. Returns passages with ids, or REFUSED when evidence is too thin."""
        return tools_for_caller(query).search_manuals(query)

    @server.tool()
    def run_sql(sql: str) -> str:
        """Run one read-only Snowflake SELECT as the caller. Returns rows as passages, or REFUSED with the reason."""
        return tools_for_caller(sql).run_sql(sql)

    return server


if __name__ == "__main__":
    from runtime import build_agent_deps

    build_server(build_agent_deps()).run(transport="streamable-http")
