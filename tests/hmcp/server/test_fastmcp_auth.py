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
    app = Mock()
    middleware = AuthMiddleware(app, mock_auth_config)
    # Replace the JWT handler with our mock
    middleware.jwt_handler = mock_jwt_handler
    return middleware


@pytest.mark.asyncio
async def test_oauth_endpoint_bypass(auth_middleware, mock_auth_config):
    """Test that OAuth endpoints bypass authentication"""
    mock_request = Mock(spec=Request)
    mock_request.url.path = mock_auth_config.OAUTH_TOKEN_URL

    mock_call_next = AsyncMock()
    mock_call_next.return_value = "response"

    result = await auth_middleware.dispatch(mock_request, mock_call_next)

    assert result == "response"
    mock_call_next.assert_called_once_with(mock_request)


@pytest.mark.asyncio
async def test_missing_auth_header(auth_middleware):
    """Test request with missing authorization header"""
    mock_request = Mock(spec=Request)
    mock_request.url.path = "/api/endpoint"
    mock_request.headers = {}

    result = await auth_middleware.dispatch(mock_request, AsyncMock())

    assert isinstance(result, JSONResponse)
    assert result.status_code == 401
    assert result.body == b'{"error":"No authorization header provided"}'


@pytest.mark.asyncio
async def test_successful_authentication(auth_middleware, mock_utils, mock_jwt_handler):
    """Test successful authentication flow"""
    # Mock request
    mock_request = Mock(spec=Request)
    mock_request.url.path = "/api/endpoint"
    mock_request.headers = {"Authorization": "Bearer test_token"}
    mock_request.state = Mock()

    # Mock token parsing and verification
    mock_utils.parse_auth_header.return_value = "parsed_token"
    mock_jwt_handler.verify_token.return_value = {
        "sub": "test_client",
        "scope": "patient/read patient/write",
        "patient": "123",
    }

    # Mock next middleware
    mock_call_next = AsyncMock()
    mock_call_next.return_value = "success_response"

    result = await auth_middleware.dispatch(mock_request, mock_call_next)

    # Verify the result
    assert result == "success_response"

    # Verify token parsing and verification
    mock_utils.parse_auth_header.assert_called_once_with("Bearer test_token")
    mock_jwt_handler.verify_token.assert_called_once_with("parsed_token")

    # Verify request state was updated
    assert mock_request.state.client_id == "test_client"
    assert mock_request.state.scopes == ["patient/read", "patient/write"]
    assert mock_request.state.patient_id == "123"


@pytest.mark.asyncio
async def test_invalid_client_id(auth_middleware, mock_utils, mock_jwt_handler):
    """Test authentication with invalid client ID"""
    mock_request = Mock(spec=Request)
    mock_request.url.path = "/api/endpoint"
    mock_request.headers = {"Authorization": "Bearer test_token"}

    mock_utils.parse_auth_header.return_value = "parsed_token"
    mock_jwt_handler.verify_token.return_value = {
        "sub": "invalid_client",
        "scope": "patient/read",
    }

    result = await auth_middleware.dispatch(mock_request, AsyncMock())

    assert isinstance(result, JSONResponse)
    assert result.status_code == 401
    assert result.body == b'{"error":"Invalid client ID"}'


@pytest.mark.asyncio
async def test_insufficient_scope(auth_middleware, mock_utils, mock_jwt_handler):
    """Test authentication with insufficient scopes"""
    mock_request = Mock(spec=Request)
    mock_request.url.path = "/api/endpoint"
    mock_request.headers = {"Authorization": "Bearer test_token"}

    mock_utils.parse_auth_header.return_value = "parsed_token"
    mock_jwt_handler.verify_token.return_value = {
        "sub": "test_client",
        "scope": "patient/read",  # Missing patient/write scope
    }

    result = await auth_middleware.dispatch(mock_request, AsyncMock())

    assert isinstance(result, JSONResponse)
    assert result.status_code == 403
    assert result.body == b'{"error":"Insufficient scope"}'


@pytest.mark.asyncio
async def test_missing_patient_id(auth_middleware, mock_utils, mock_jwt_handler):
    """Test patient-context scopes without patient ID"""
    mock_request = Mock(spec=Request)
    mock_request.url.path = "/api/endpoint"
    mock_request.headers = {"Authorization": "Bearer test_token"}

    mock_utils.parse_auth_header.return_value = "parsed_token"
    mock_jwt_handler.verify_token.return_value = {
        "sub": "test_client",
        "scope": "patient/read patient/write",  # Patient scopes without patient ID
    }

    result = await auth_middleware.dispatch(mock_request, AsyncMock())

    assert isinstance(result, JSONResponse)
    assert result.status_code == 403
    assert result.body == b'{"error":"Patient-context scopes require patient ID"}'


@pytest.mark.asyncio
async def test_invalid_token_error(auth_middleware, mock_utils, mock_jwt_handler):
    """Test handling of InvalidTokenError"""
    mock_request = Mock(spec=Request)
    mock_request.url.path = "/api/endpoint"
    mock_request.headers = {"Authorization": "Bearer test_token"}

    mock_utils.parse_auth_header.return_value = "parsed_token"
    mock_jwt_handler.verify_token.side_effect = InvalidTokenError("Token expired")

    result = await auth_middleware.dispatch(mock_request, AsyncMock())

    assert isinstance(result, JSONResponse)
    assert result.status_code == 401
    assert result.body == b'{"error":"Authentication failed: Token expired"}'


@pytest.mark.asyncio
async def test_scope_error(auth_middleware, mock_utils, mock_jwt_handler):
    """Test handling of ScopeError"""
    mock_request = Mock(spec=Request)
    mock_request.url.path = "/api/endpoint"
    mock_request.headers = {"Authorization": "Bearer test_token"}

    mock_utils.parse_auth_header.return_value = "parsed_token"
    mock_jwt_handler.verify_token.side_effect = ScopeError("Invalid scope format")

    result = await auth_middleware.dispatch(mock_request, AsyncMock())

    assert isinstance(result, JSONResponse)
    assert result.status_code == 403
    assert result.body == b'{"error":"Authorization failed: Invalid scope format"}'


@pytest.mark.asyncio
async def test_general_exception(auth_middleware, mock_utils, mock_jwt_handler):
    """Test handling of general exceptions"""
    mock_request = Mock(spec=Request)
    mock_request.url.path = "/api/endpoint"
    mock_request.headers = {"Authorization": "Bearer test_token"}

    mock_utils.parse_auth_header.return_value = "parsed_token"
    mock_jwt_handler.verify_token.side_effect = Exception("Unexpected error")

    result = await auth_middleware.dispatch(mock_request, AsyncMock())

    assert isinstance(result, JSONResponse)
    assert result.status_code == 401
    assert result.body == b'{"error":"Authentication failed: Unexpected error"}'


@pytest.mark.asyncio
async def test_empty_scope_string(auth_middleware, mock_utils, mock_jwt_handler):
    """Test authentication with empty scope string"""
    mock_request = Mock(spec=Request)
    mock_request.url.path = "/api/endpoint"
    mock_request.headers = {"Authorization": "Bearer test_token"}

    mock_utils.parse_auth_header.return_value = "parsed_token"
    mock_jwt_handler.verify_token.return_value = {
        "sub": "test_client",
        "scope": "",  # Empty scope string
    }

    result = await auth_middleware.dispatch(mock_request, AsyncMock())

    assert isinstance(result, JSONResponse)
    assert result.status_code == 403
    assert result.body == b'{"error":"Insufficient scope"}'


@pytest.mark.asyncio
async def test_malformed_auth_header(auth_middleware, mock_utils):
    """Test handling of malformed authorization header"""
    mock_request = Mock(spec=Request)
    mock_request.url.path = "/api/endpoint"
    mock_request.headers = {"Authorization": "InvalidFormat"}

    mock_utils.parse_auth_header.side_effect = InvalidTokenError(
        "Invalid authorization header format"
    )

    result = await auth_middleware.dispatch(mock_request, AsyncMock())

    assert isinstance(result, JSONResponse)
    assert result.status_code == 401
    assert (
        result.body
        == b'{"error":"Authentication failed: Invalid authorization header format"}'
    )


@pytest.mark.asyncio
async def test_multiple_patient_ids(auth_middleware, mock_utils, mock_jwt_handler):
    """Test authentication with multiple patient IDs"""
    mock_request = Mock(spec=Request)
    mock_request.url.path = "/api/endpoint"
    mock_request.headers = {"Authorization": "Bearer test_token"}
    mock_request.state = Mock()

    mock_utils.parse_auth_header.return_value = "parsed_token"
    mock_jwt_handler.verify_token.return_value = {
        "sub": "test_client",
        "scope": "patient/read patient/write",
        "patient": ["123", "456"],  # Multiple patient IDs
    }

    mock_call_next = AsyncMock()
    mock_call_next.return_value = "success_response"

    result = await auth_middleware.dispatch(mock_request, mock_call_next)

    assert result == "success_response"
    assert mock_request.state.client_id == "test_client"
    assert mock_request.state.scopes == ["patient/read", "patient/write"]
    assert mock_request.state.patient_id == ["123", "456"]
