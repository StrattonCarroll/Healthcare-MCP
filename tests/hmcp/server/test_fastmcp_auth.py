import pytest
from unittest.mock import Mock, patch, AsyncMock
from starlette.requests import Request
from starlette.responses import JSONResponse
from hmcp.shared.auth import AuthConfig, InvalidTokenError, ScopeError
from hmcp.server.fastmcp_auth import AuthMiddleware


@pytest.fixture
def mock_auth_config():
    """Mock AuthConfig with test settings"""
    config = AuthConfig()
    config.OAUTH_TOKEN_URL = "/oauth/token"
    config.ALLOWED_CLIENTS = {
        "test_client": {"scopes": ["patient/read", "patient/write"]}
    }
    return config


@pytest.fixture
def mock_jwt_handler():
    """Mock JWTHandler"""
    with patch("hmcp.server.fastmcp_auth.jwt_handler.JWTHandler") as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_utils():
    """Mock auth utils"""
    with patch("hmcp.server.fastmcp_auth.utils") as mock:
        yield mock


@pytest.fixture
def auth_middleware(mock_auth_config, mock_jwt_handler):
    """Create AuthMiddleware instance with mocked config"""
    app = AsyncMock()
    middleware = AuthMiddleware(app, mock_auth_config)
    # Replace the JWT handler with our mock
    middleware.jwt_handler = mock_jwt_handler
    return middleware


@pytest.fixture
def mock_scope():
    """Create a mock ASGI scope"""
    return {
        "type": "http",
        "method": "GET",
        "path": "/api/endpoint",
        "headers": [],
        "state": {}
    }


@pytest.fixture
def mock_receive():
    """Create a mock ASGI receive function"""
    return AsyncMock()


@pytest.fixture
def mock_send():
    """Create a mock ASGI send function"""
    return AsyncMock()


@pytest.mark.asyncio
async def test_oauth_endpoint_bypass(auth_middleware, mock_auth_config, mock_scope, mock_receive, mock_send):
    """Test that OAuth endpoints bypass authentication"""
    mock_scope["path"] = mock_auth_config.OAUTH_TOKEN_URL

    await auth_middleware(mock_scope, mock_receive, mock_send)

    auth_middleware.app.assert_called_once_with(
        mock_scope, mock_receive, mock_send)


@pytest.mark.asyncio
async def test_missing_auth_header(auth_middleware, mock_scope, mock_receive, mock_send):
    """Test request with missing authorization header"""
    mock_scope["headers"] = []

    await auth_middleware(mock_scope, mock_receive, mock_send)

    # Verify both start and body messages were sent
    assert mock_send.call_count == 2

    # Check start message
    start_call = mock_send.call_args_list[0][0][0]
    assert start_call["type"] == "http.response.start"
    assert start_call["status"] == 401
    assert start_call["headers"] == [(b"content-type", b"application/json")]

    # Check body message
    body_call = mock_send.call_args_list[1][0][0]
    assert body_call["type"] == "http.response.body"
    assert body_call["body"] == b'{"error":"No authorization header provided"}'


@pytest.mark.asyncio
async def test_successful_authentication(auth_middleware, mock_utils, mock_jwt_handler, mock_scope, mock_receive, mock_send):
    """Test successful authentication flow"""
    # Setup scope with auth header
    mock_scope["headers"] = [(b"authorization", b"Bearer test_token")]
    mock_scope["state"] = {}

    # Mock token parsing and verification
    mock_utils.parse_auth_header.return_value = "parsed_token"
    mock_jwt_handler.verify_token.return_value = {
        "sub": "test_client",
        "scope": "patient/read patient/write",
        "patient": "123",
    }

    await auth_middleware(mock_scope, mock_receive, mock_send)

    # Verify token parsing and verification
    mock_utils.parse_auth_header.assert_called_once_with("Bearer test_token")
    mock_jwt_handler.verify_token.assert_called_once_with("parsed_token")

    # Verify request state was updated
    assert mock_scope["state"]["client_id"] == "test_client"
    assert mock_scope["state"]["scopes"] == ["patient/read", "patient/write"]
    assert mock_scope["state"]["patient_id"] == "123"

    # Verify app was called
    auth_middleware.app.assert_called_once_with(
        mock_scope, mock_receive, mock_send)


@pytest.mark.asyncio
async def test_invalid_client_id(auth_middleware, mock_utils, mock_jwt_handler, mock_scope, mock_receive, mock_send):
    """Test authentication with invalid client ID"""
    mock_scope["headers"] = [(b"authorization", b"Bearer test_token")]

    mock_utils.parse_auth_header.return_value = "parsed_token"
    mock_jwt_handler.verify_token.return_value = {
        "sub": "invalid_client",
        "scope": "patient/read",
    }

    await auth_middleware(mock_scope, mock_receive, mock_send)

    # Verify both start and body messages were sent
    assert mock_send.call_count == 2

    # Check start message
    start_call = mock_send.call_args_list[0][0][0]
    assert start_call["type"] == "http.response.start"
    assert start_call["status"] == 401
    assert start_call["headers"] == [(b"content-type", b"application/json")]

    # Check body message
    body_call = mock_send.call_args_list[1][0][0]
    assert body_call["type"] == "http.response.body"
    assert body_call["body"] == b'{"error":"Invalid client ID"}'


@pytest.mark.asyncio
async def test_insufficient_scope(auth_middleware, mock_utils, mock_jwt_handler, mock_scope, mock_receive, mock_send):
    """Test authentication with insufficient scopes"""
    mock_scope["headers"] = [(b"authorization", b"Bearer test_token")]

    mock_utils.parse_auth_header.return_value = "parsed_token"
    mock_jwt_handler.verify_token.return_value = {
        "sub": "test_client",
        "scope": "patient/read",  # Missing patient/write scope
    }

    await auth_middleware(mock_scope, mock_receive, mock_send)

    # Verify both start and body messages were sent
    assert mock_send.call_count == 2

    # Check start message
    start_call = mock_send.call_args_list[0][0][0]
    assert start_call["type"] == "http.response.start"
    assert start_call["status"] == 403
    assert start_call["headers"] == [(b"content-type", b"application/json")]

    # Check body message
    body_call = mock_send.call_args_list[1][0][0]
    assert body_call["type"] == "http.response.body"
    assert body_call["body"] == b'{"error":"Insufficient scope"}'


@pytest.mark.asyncio
async def test_missing_patient_id(auth_middleware, mock_utils, mock_jwt_handler, mock_scope, mock_receive, mock_send):
    """Test patient-context scopes without patient ID"""
    mock_scope["headers"] = [(b"authorization", b"Bearer test_token")]

    mock_utils.parse_auth_header.return_value = "parsed_token"
    mock_jwt_handler.verify_token.return_value = {
        "sub": "test_client",
        "scope": "patient/read patient/write",  # Patient scopes without patient ID
    }

    await auth_middleware(mock_scope, mock_receive, mock_send)

    # Verify both start and body messages were sent
    assert mock_send.call_count == 2

    # Check start message
    start_call = mock_send.call_args_list[0][0][0]
    assert start_call["type"] == "http.response.start"
    assert start_call["status"] == 403
    assert start_call["headers"] == [(b"content-type", b"application/json")]

    # Check body message
    body_call = mock_send.call_args_list[1][0][0]
    assert body_call["type"] == "http.response.body"
    assert body_call["body"] == b'{"error":"Patient-context scopes require patient ID"}'


@pytest.mark.asyncio
async def test_invalid_token_error(auth_middleware, mock_utils, mock_jwt_handler, mock_scope, mock_receive, mock_send):
    """Test handling of InvalidTokenError"""
    mock_scope["headers"] = [(b"authorization", b"Bearer test_token")]

    mock_utils.parse_auth_header.return_value = "parsed_token"
    mock_jwt_handler.verify_token.side_effect = InvalidTokenError(
        "Token expired")

    await auth_middleware(mock_scope, mock_receive, mock_send)

    # Verify both start and body messages were sent
    assert mock_send.call_count == 2

    # Check start message
    start_call = mock_send.call_args_list[0][0][0]
    assert start_call["type"] == "http.response.start"
    assert start_call["status"] == 401
    assert start_call["headers"] == [(b"content-type", b"application/json")]

    # Check body message
    body_call = mock_send.call_args_list[1][0][0]
    assert body_call["type"] == "http.response.body"
    assert body_call["body"] == b'{"error":"Authentication failed: Token expired"}'


@pytest.mark.asyncio
async def test_scope_error(auth_middleware, mock_utils, mock_jwt_handler, mock_scope, mock_receive, mock_send):
    """Test handling of ScopeError"""
    mock_scope["headers"] = [(b"authorization", b"Bearer test_token")]

    mock_utils.parse_auth_header.return_value = "parsed_token"
    mock_jwt_handler.verify_token.side_effect = ScopeError(
        "Invalid scope format")

    await auth_middleware(mock_scope, mock_receive, mock_send)

    # Verify both start and body messages were sent
    assert mock_send.call_count == 2

    # Check start message
    start_call = mock_send.call_args_list[0][0][0]
    assert start_call["type"] == "http.response.start"
    assert start_call["status"] == 403
    assert start_call["headers"] == [(b"content-type", b"application/json")]

    # Check body message
    body_call = mock_send.call_args_list[1][0][0]
    assert body_call["type"] == "http.response.body"
    assert body_call["body"] == b'{"error":"Authorization failed: Invalid scope format"}'


@pytest.mark.asyncio
async def test_general_exception(auth_middleware, mock_utils, mock_jwt_handler, mock_scope, mock_receive, mock_send):
    """Test handling of general exceptions"""
    mock_scope["headers"] = [(b"authorization", b"Bearer test_token")]

    mock_utils.parse_auth_header.return_value = "parsed_token"
    mock_jwt_handler.verify_token.side_effect = Exception("Unexpected error")

    await auth_middleware(mock_scope, mock_receive, mock_send)

    # Verify both start and body messages were sent
    assert mock_send.call_count == 2

    # Check start message
    start_call = mock_send.call_args_list[0][0][0]
    assert start_call["type"] == "http.response.start"
    assert start_call["status"] == 401
    assert start_call["headers"] == [(b"content-type", b"application/json")]

    # Check body message
    body_call = mock_send.call_args_list[1][0][0]
    assert body_call["type"] == "http.response.body"
    assert body_call["body"] == b'{"error":"Authentication failed: Unexpected error"}'


@pytest.mark.asyncio
async def test_empty_scope_string(auth_middleware, mock_utils, mock_jwt_handler, mock_scope, mock_receive, mock_send):
    """Test authentication with empty scope string"""
    mock_scope["headers"] = [(b"authorization", b"Bearer test_token")]

    mock_utils.parse_auth_header.return_value = "parsed_token"
    mock_jwt_handler.verify_token.return_value = {
        "sub": "test_client",
        "scope": "",  # Empty scope string
    }

    await auth_middleware(mock_scope, mock_receive, mock_send)

    # Verify both start and body messages were sent
    assert mock_send.call_count == 2

    # Check start message
    start_call = mock_send.call_args_list[0][0][0]
    assert start_call["type"] == "http.response.start"
    assert start_call["status"] == 403
    assert start_call["headers"] == [(b"content-type", b"application/json")]

    # Check body message
    body_call = mock_send.call_args_list[1][0][0]
    assert body_call["type"] == "http.response.body"
    assert body_call["body"] == b'{"error":"Insufficient scope"}'


@pytest.mark.asyncio
async def test_malformed_auth_header(auth_middleware, mock_utils, mock_scope, mock_receive, mock_send):
    """Test handling of malformed authorization header"""
    mock_scope["headers"] = [(b"authorization", b"InvalidFormat")]

    mock_utils.parse_auth_header.side_effect = InvalidTokenError(
        "Invalid authorization header format"
    )

    await auth_middleware(mock_scope, mock_receive, mock_send)

    # Verify both start and body messages were sent
    assert mock_send.call_count == 2

    # Check start message
    start_call = mock_send.call_args_list[0][0][0]
    assert start_call["type"] == "http.response.start"
    assert start_call["status"] == 401
    assert start_call["headers"] == [(b"content-type", b"application/json")]

    # Check body message
    body_call = mock_send.call_args_list[1][0][0]
    assert body_call["type"] == "http.response.body"
    assert body_call["body"] == b'{"error":"Authentication failed: Invalid authorization header format"}'


@pytest.mark.asyncio
async def test_multiple_patient_ids(auth_middleware, mock_utils, mock_jwt_handler, mock_scope, mock_receive, mock_send):
    """Test authentication with multiple patient IDs"""
    mock_scope["headers"] = [(b"authorization", b"Bearer test_token")]
    mock_scope["state"] = {}

    mock_utils.parse_auth_header.return_value = "parsed_token"
    mock_jwt_handler.verify_token.return_value = {
        "sub": "test_client",
        "scope": "patient/read patient/write",
        "patient": ["123", "456"],  # Multiple patient IDs
    }

    await auth_middleware(mock_scope, mock_receive, mock_send)

    # Verify request state was updated
    assert mock_scope["state"]["client_id"] == "test_client"
    assert mock_scope["state"]["scopes"] == ["patient/read", "patient/write"]
    assert mock_scope["state"]["patient_id"] == ["123", "456"]

    # Verify app was called
    auth_middleware.app.assert_called_once_with(
        mock_scope, mock_receive, mock_send)
