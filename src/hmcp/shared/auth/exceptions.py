class AuthenticationError(Exception):
    """Base authentication error"""

    pass


class InvalidTokenError(AuthenticationError):
    """Invalid or expired token"""

    pass


class ClientValidationError(AuthenticationError):
    """Invalid client credentials"""

    pass


class ScopeError(AuthenticationError):
    """Invalid or insufficient scope"""

    pass
