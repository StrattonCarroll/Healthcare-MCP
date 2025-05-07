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
import logging

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for HMCP server

    Implements JWT Bearer token authentication with scope-based access control.
    Supports patient-context restrictions as defined in SMART on FHIR.
    """

    def __init__(self, app, auth_config: AuthConfig):
        logger.info("Initializing AuthMiddleware")
        super().__init__(app)
        self.auth_config = auth_config
        self.jwt_handler = jwt_handler.JWTHandler(auth_config)

    async def dispatch(self, request: Request, call_next):
        """Middleware to authenticate incoming requests"""
        # Skip authentication for OAuth endpoints
        if request.url.path == self.auth_config.OAUTH_TOKEN_URL:
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header:
            logger.error("No authorization header provided")
            return JSONResponse(
                content={"error": "No authorization header provided"}, status_code=401
            )

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
                return JSONResponse(
                    content={"error": "Invalid client ID"}, status_code=401
                )

            # Validate scopes
            token_scopes = payload.get("scope", "").split()
            required_scopes = self.auth_config.ALLOWED_CLIENTS[client_id].get(
                "scopes", []
            )
            if not all(scope in token_scopes for scope in required_scopes):
                logger.error(
                    f"Invalid scopes. Required: {required_scopes}, Got: {token_scopes}"
                )
                return JSONResponse(
                    content={"error": "Insufficient scope"},
                    status_code=403,  # Use 403 Forbidden for scope issues
                )

            # Process patient context if present
            patient_id = payload.get("patient")
            has_patient_scopes = any(
                scope.startswith("patient/") for scope in token_scopes
            )

            if has_patient_scopes and not patient_id:
                logger.error(
                    "Patient-context scopes requested but no patient ID in token"
                )
                return JSONResponse(
                    content={"error": "Patient-context scopes require patient ID"},
                    status_code=403,
                )

            logger.info(f"Authentication successful for client: {client_id}")
            # Add the authenticated client info to the request state
            request.state.client_id = client_id
            request.state.scopes = token_scopes
            if patient_id:
                request.state.patient_id = patient_id

            return await call_next(request)
        except InvalidTokenError as e:
            logger.error(
                f"Authentication failed: Invalid Token - {str(e)}", exc_info=False
            )
            return JSONResponse(
                content={"error": f"Authentication failed: {str(e)}"}, status_code=401
            )
        except ScopeError as e:
            logger.error(
                f"Authorization failed: Scope error - {str(e)}", exc_info=False
            )
            return JSONResponse(
                content={"error": f"Authorization failed: {str(e)}"}, status_code=403
            )
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}", exc_info=True)
            return JSONResponse(
                content={"error": f"Authentication failed: {str(e)}"}, status_code=401
            )
