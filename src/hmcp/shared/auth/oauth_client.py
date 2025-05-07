from typing import Optional, Dict, List
from .exceptions import AuthenticationError, ClientValidationError
from .config import AuthConfig


class OAuthClient:
    """OAuth 2.0 client implementation supporting SMART on FHIR authentication flows.

    Current implementation supports Client Credentials flow only.
    """

    def __init__(self, client_id: str, client_secret: str, config: AuthConfig):
        self.client_id = client_id
        self.client_secret = client_secret
        self.config = config
        self.access_token: Optional[str] = None
        self.id_token: Optional[str] = None  # For OpenID Connect (future)
        self.token_response: Optional[Dict] = None  # Store full token response
        self.patient_id: Optional[str] = None  # For patient-context operations
        self.validate_client()

    def validate_client(self):
        """Validate client credentials"""
        if self.client_id not in self.config.ALLOWED_CLIENTS:
            raise ClientValidationError(f"Client {self.client_id} is not registered")
        if self.config.ALLOWED_CLIENTS[self.client_id]["secret"] != self.client_secret:
            raise ClientValidationError("Invalid client secret")

    def create_token_request(
        self, scopes: List[str] = None, patient_id: str = None
    ) -> dict:
        """Create token request payload for Client Credentials flow

        Args:
            scopes: List of requested scopes (defaults to config scopes)
            patient_id: Optional patient identifier for patient-context operations
        """
        requested_scopes = scopes or self.config.OAUTH_SCOPES

        # Add patient context scope if patient_id is provided
        if patient_id and not any(s.startswith("patient/") for s in requested_scopes):
            requested_scopes = list(requested_scopes)  # Convert to list if it's a tuple
            requested_scopes.append("launch/patient")

        request = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials",
            "scope": " ".join(requested_scopes),
        }

        return request

    def set_token(self, token_response: dict):
        """Process and store token response

        Stores the full token response, including access_token, id_token (if OpenID Connect),
        and patient context (if available).
        """
        self.token_response = token_response
        self.access_token = token_response["access_token"]

        # Store ID token if present (OpenID Connect)
        if "id_token" in token_response:
            self.id_token = token_response["id_token"]

        # Store patient ID if present
        if "patient" in token_response:
            self.patient_id = token_response["patient"]

    def get_auth_header(self) -> dict:
        """Get authorization header for requests"""
        if not self.access_token:
            raise AuthenticationError("Not authenticated")
        return {"Authorization": f"Bearer {self.access_token}"}
