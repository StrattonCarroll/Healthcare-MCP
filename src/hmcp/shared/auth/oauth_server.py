from datetime import datetime
from typing import Dict, Optional, List
from .jwt_handler import JWTHandler
from .exceptions import ClientValidationError, AuthenticationError
from .config import AuthConfig


class OAuthServer:
    """OAuth 2.0 server implementation supporting Client Credentials and Authorization Code flows.

    Current implementation supports Client Credentials flow only.
    """

    def __init__(self, config: AuthConfig):
        self.config = config
        self.jwt_handler = JWTHandler(config)
        # Replace with database in production
        self.oauth_clients: Dict[str, Dict] = {}

    def register_client(
        self,
        client_id: str,
        client_secret: str,
        redirect_uris: List[str] = None,
        allowed_scopes: List[str] = None,
    ):
        """Register a new OAuth client

        Args:
            client_id: Client identifier
            client_secret: Client secret
            redirect_uris: List of allowed redirect URIs for Authorization Code flow
            allowed_scopes: List of allowed scopes for this client
        """
        self.oauth_clients[client_id] = {
            "secret": client_secret,
            "created_at": datetime.utcnow(),
            "redirect_uris": redirect_uris or [],
            "allowed_scopes": allowed_scopes or ["hmcp:access"],
        }

    def validate_client(self, client_id: str, client_secret: str) -> bool:
        """Validate client credentials"""
        if client_id not in self.oauth_clients:
            raise ClientValidationError("Invalid client ID")
        return self.oauth_clients[client_id]["secret"] == client_secret

    def create_token(self, client_id: str, scope: str, patient_id: str = None) -> Dict:
        """Create OAuth token response for Client Credentials flow

        Args:
            client_id: Client identifier
            scope: Space-separated list of requested scopes
            patient_id: Optional patient identifier for patient-context scopes

        Returns:
            Dictionary containing token response with access_token, token_type, etc.
        """
        token = self.jwt_handler.generate_token(client_id, scope, patient_id)

        response = {
            "access_token": token,
            "token_type": "Bearer",
            "expires_in": self.config.TOKEN_EXPIRY_HOURS * 3600,
            "scope": scope,
        }

        # Add patient claim to response if patient_id is provided
        if patient_id:
            response["patient"] = patient_id

        return response

    # Authorization Code flow methods (placeholder for future implementation)
    def create_authorization_code(
        self, client_id: str, redirect_uri: str, scope: str, state: str
    ) -> str:
        """Create and store an authorization code for Authorization Code flow

        Note: Not yet implemented - placeholder for future development
        """
        # This would generate a short-lived code and store it with associated data
        raise NotImplementedError("Authorization Code flow is not yet implemented")

    def exchange_code_for_token(
        self, client_id: str, client_secret: str, code: str, redirect_uri: str
    ) -> Dict:
        """Exchange authorization code for token in Authorization Code flow

        Note: Not yet implemented - placeholder for future development
        """
        # This would validate the code and client, then issue tokens
        raise NotImplementedError("Authorization Code flow is not yet implemented")

    # OpenID Connect features (placeholder for future implementation)
    def create_id_token(self, client_id: str, user_id: str) -> str:
        """Create an OpenID Connect ID Token

        Note: Not yet implemented - placeholder for future development
        """
        raise NotImplementedError("OpenID Connect is not yet implemented")
