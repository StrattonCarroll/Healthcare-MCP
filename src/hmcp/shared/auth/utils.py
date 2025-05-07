import secrets
from typing import List, Tuple, Set


def generate_client_secret() -> str:
    """Generate secure client secret"""
    return secrets.token_urlsafe(32)


def validate_scope(required_scope: str, token_scope: str) -> bool:
    """Validate if token has required scope"""
    token_scopes = set(token_scope.split())
    return required_scope in token_scopes


def parse_auth_header(auth_header: str) -> str:
    """Parse Bearer token from Authorization header"""
    if not auth_header or not auth_header.startswith("Bearer "):
        raise ValueError("Invalid Authorization header")
    return auth_header.split(" ")[1]


def analyze_scopes(scopes: List[str]) -> Tuple[Set[str], bool, bool]:
    """Analyze scopes to identify their types

    Args:
        scopes: List of scopes to analyze

    Returns:
        Tuple containing:
        - Set of resource scopes (non-system scopes)
        - Boolean indicating if patient-context scopes are present
        - Boolean indicating if OpenID Connect scopes are present
    """
    resource_scopes = set()
    has_patient_context = False
    has_openid = False

    for scope in scopes:
        if scope == "openid":
            has_openid = True
        elif scope.startswith("patient/"):
            has_patient_context = True
            resource_scopes.add(scope[8:])  # Strip "patient/" prefix
        elif scope not in ["profile", "offline_access", "launch/patient"]:
            resource_scopes.add(scope)

    # Also check for launch/patient scope which indicates patient context
    if "launch/patient" in scopes:
        has_patient_context = True

    return resource_scopes, has_patient_context, has_openid


def requires_patient_context(scopes: List[str]) -> bool:
    """Check if the requested scopes require patient context

    Args:
        scopes: List of scopes to check

    Returns:
        True if patient-context scopes are present
    """
    return (
        any(scope.startswith("patient/") for scope in scopes)
        or "launch/patient" in scopes
    )
