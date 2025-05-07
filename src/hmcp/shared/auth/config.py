from dataclasses import dataclass, field
from typing import List


@dataclass
class AuthConfig:
    # Load from environment in production
    JWT_SECRET_KEY: str = "your-secure-jwt-secret"
    JWT_ALGORITHM: str = "HS256"
    TOKEN_EXPIRY_HOURS: int = 1
    ISSUER: str = "HMCP_Server"
    # Audience claim for JWT - should identify the recipient server
    JWT_AUDIENCE: str = "https://hmcp-server.example.com"

    # OAuth settings
    OAUTH_TOKEN_URL: str = "/oauth/token"
    # Updated to include all scopes defined in the documentation
    OAUTH_SCOPES: List[str] = field(
        default_factory=lambda: [
            "hmcp:access",
            "hmcp:read",
            "hmcp:write",
            "patient/hmcp:read",
            "patient/hmcp:write",
            "openid",
            "profile",
            "launch/patient",
            "offline_access",
        ]
    )

    # Other config fields should also use field() if they have mutable defaults
    # Removing redundant JWT_SECRET as JWT_SECRET_KEY is the one used
    TOKEN_EXPIRY: int = 3600  # 1 hour in seconds
    ALLOWED_CLIENTS: dict = field(
        default_factory=lambda: {
            "test-client": {"secret": "test-secret", "scopes": ["hmcp:access"]}
        }
    )
