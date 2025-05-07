import jwt as pyjwt
import logging
from datetime import datetime, timedelta
from .config import AuthConfig
from .exceptions import InvalidTokenError, ClientValidationError

logger = logging.getLogger(__name__)


class JWTHandler:
    def __init__(self, config: AuthConfig):
        self.config = config
        self.blacklisted_tokens = set()

    def generate_token(self, client_id: str, scope: str, patient_id: str = None) -> str:
        """Generate JWT token with claims

        Args:
            client_id: The client identifier
            scope: Space-separated list of requested scopes
            patient_id: Optional patient identifier for patient-context scopes

        Returns:
            JWT token string

        Raises:
            ClientValidationError: If client_id is not in allowed clients
        """
        logger.debug(
            f"Generating token for client_id: {client_id}, scope: {scope}, patient_id: {patient_id}"
        )

        # Validate client_id against allowed clients
        if client_id not in self.config.ALLOWED_CLIENTS:
            logger.error(f"Invalid client_id: {client_id}")
            raise ClientValidationError(f"Client {client_id} is not registered")

        now = datetime.utcnow()
        payload = {
            "iss": self.config.ISSUER,
            "sub": client_id,
            "aud": self.config.JWT_AUDIENCE,  # Add audience claim
            "iat": now,
            "exp": now + timedelta(hours=self.config.TOKEN_EXPIRY_HOURS),
            "scope": scope,
        }

        # Add patient claim if patient-context scopes are requested and patient_id is provided
        if patient_id and any(scope.startswith("patient/") for scope in scope.split()):
            payload["patient"] = patient_id

        logger.debug(f"Token payload: {payload}")

        try:
            token = pyjwt.encode(
                payload, self.config.JWT_SECRET_KEY, algorithm=self.config.JWT_ALGORITHM
            )
            return token
        except Exception as e:
            logger.error(f"Failed to generate token: {str(e)}")
            raise

    def verify_token(self, token: str) -> dict:
        """Verify JWT token and return payload

        Args:
            token: JWT token string to verify

        Returns:
            JWT payload as dictionary

        Raises:
            InvalidTokenError: If token is invalid, expired, or revoked
        """
        if token in self.blacklisted_tokens:
            logger.error("Token has been revoked")
            raise InvalidTokenError("Token has been revoked")

        try:
            payload = pyjwt.decode(
                token,
                self.config.JWT_SECRET_KEY,
                algorithms=[self.config.JWT_ALGORITHM],
                audience=self.config.JWT_AUDIENCE,  # Validate audience claim
            )
            return payload
        except pyjwt.ExpiredSignatureError:
            logger.error("Token has expired")
            raise InvalidTokenError("Token has expired")
        except pyjwt.InvalidAudienceError:
            logger.error("Invalid token audience")
            raise InvalidTokenError("Invalid token audience")
        except pyjwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {str(e)}")
            raise InvalidTokenError("Invalid token")
        except Exception as e:
            logger.error(
                f"Unexpected error during token verification: {str(e)}", exc_info=True
            )
            raise InvalidTokenError(f"Token verification failed: {str(e)}")
