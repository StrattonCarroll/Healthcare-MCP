from typing import Any, Optional
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from mcp.server.sse import SseServerTransport
from hmcp.shared.auth import (
    OAuthServer,
    AuthConfig,
    jwt_handler,
    utils,
    InvalidTokenError,
    ScopeError,
)
from starlette.authentication import AuthenticationError
from starlette.requests import HTTPConnection
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send
import logging
import typing

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for HMCP server

    Implements JWT Bearer token authentication with scope-based access control.
    Supports patient-context restrictions as defined in SMART on FHIR.
    """

    def __init__(
        self,
        app: ASGIApp,
        auth_config: AuthConfig,
        on_error: (
            typing.Callable[[HTTPConnection, AuthenticationError], Response] | None
        ) = None,
    ) -> None:
        self.app = app
        self.auth_config = auth_config
        self.jwt_handler = jwt_handler.JWTHandler(auth_config)
        self.on_error: typing.Callable[
            [HTTPConnection, AuthenticationError], Response
        ] = (on_error if on_error is not None else self.default_on_error)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Middleware to authenticate incoming requests"""
        # Skip authentication for OAuth endpoints
        request = Request(scope, receive, send)
        if request.url.path == self.auth_config.OAUTH_TOKEN_URL:
            await self.app(scope, receive, send)
            return

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            logger.error("No authorization header provided")
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"error":"No authorization header provided"}',
                }
            )
            return

        try:
            logger.debug(
                f"Authentication for {request.method} {request.url.path} - {auth_header[:15]}..."
            )
            token = utils.parse_auth_header(auth_header)

            payload = self.jwt_handler.verify_token(token)

            # Validate client ID
            client_id = payload.get("sub")
            if not client_id or client_id not in self.auth_config.ALLOWED_CLIENTS:
                logger.error(f"Invalid client ID: {client_id}")
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"error":"Invalid client ID"}',
                    }
                )
                return

            # Validate scopes
            token_scopes = payload.get("scope", "").split()
            required_scopes = self.auth_config.ALLOWED_CLIENTS[client_id].get(
                "scopes", []
            )
            if not all(scope in token_scopes for scope in required_scopes):
                logger.error(
                    f"Invalid scopes. Required: {required_scopes}, Got: {token_scopes}"
                )
                await send(
                    {
                        "type": "http.response.start",
                        "status": 403,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"error":"Insufficient scope"}',
                    }
                )
                return

            # Process patient context if present
            patient_id = payload.get("patient")
            has_patient_scopes = any(
                scope.startswith("patient/") for scope in token_scopes
            )

            if has_patient_scopes and not patient_id:
                logger.error(
                    "Patient-context scopes requested but no patient ID in token"
                )
                await send(
                    {
                        "type": "http.response.start",
                        "status": 403,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"error":"Patient-context scopes require patient ID"}',
                    }
                )
                return

            logger.info(f"Authentication successful for client: {client_id}")
            # Add the authenticated client info to the request state
            request.state.client_id = client_id
            request.state.scopes = token_scopes
            if patient_id:
                request.state.patient_id = patient_id

            await self.app(scope, receive, send)

        except InvalidTokenError as e:
            logger.error(
                f"Authentication failed: Invalid Token - {str(e)}", exc_info=False
            )
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": f'{{"error":"Authentication failed: {str(e)}"}}'.encode(),
                }
            )
        except ScopeError as e:
            logger.error(
                f"Authorization failed: Scope error - {str(e)}", exc_info=False
            )
            await send(
                {
                    "type": "http.response.start",
                    "status": 403,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": f'{{"error":"Authorization failed: {str(e)}"}}'.encode(),
                }
            )
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}", exc_info=True)
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": f'{{"error":"Authentication failed: {str(e)}"}}'.encode(),
                }
            )

    @staticmethod
    def default_on_error(conn: HTTPConnection, exc: Exception) -> Response:
        return PlainTextResponse(str(exc), status_code=400)
