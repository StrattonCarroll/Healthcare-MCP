import os
import pytest
from unittest.mock import Mock, patch, AsyncMock, PropertyMock
from typing import Any
import mcp.types as types
from hmcp.server.hmcp_server import HMCPServer, SamplingFnT
from hmcp.shared.auth import AuthConfig
from hmcp.shared.guardrail_config.guardrail import GuardrailException, Guardrail
from contextvars import ContextVar
from mcp.shared.context import RequestContext
from mcp.server.lowlevel.server import request_ctx


@pytest.fixture
def mock_auth_config():
    return AuthConfig()


@pytest.fixture
def mock_guardrail():
    guardrail = Mock(spec=Guardrail)
    guardrail.run = AsyncMock()
    return guardrail


@pytest.fixture
def mock_openai_api_key():
    """Mock OpenAI API key environment variable"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        yield


@pytest.fixture
def basic_server(mock_auth_config):
    return HMCPServer(
        name="test_server", auth_config=mock_auth_config, enable_guardrails=False
    )


@pytest.fixture
def server_with_guardrails(mock_auth_config, mock_guardrail):
    return HMCPServer(
        name="test_server",
        auth_config=mock_auth_config,
        enable_guardrails=True,
        guardrail_instance=mock_guardrail,
    )


# Create a mock request context for testing
@pytest.fixture
def mock_request_context():
    ctx = Mock(spec=RequestContext)
    token = request_ctx.set(ctx)
    yield ctx
    request_ctx.reset(token)


@pytest.fixture
def mock_guardrail_class(mocker):
    """Mock Guardrail class to avoid OpenAI API key requirement"""
    mock = mocker.patch("hmcp.server.hmcp_server.Guardrail")
    return mock


def test_server_initialization(basic_server):
    """Test basic server initialization"""
    assert basic_server.name == "test_server"
    assert basic_server.enable_guardrails is False
    assert basic_server.guardrail is None
    assert "hmcp" in basic_server.experimentalCapabilities
    assert basic_server.experimentalCapabilities["hmcp"]["sampling"] is True


def test_server_initialization_with_guardrails(server_with_guardrails, mock_guardrail):
    """Test server initialization with guardrails enabled"""
    assert server_with_guardrails.enable_guardrails is True
    assert server_with_guardrails.guardrail == mock_guardrail
    assert server_with_guardrails.experimentalCapabilities["hmcp"]["guardrails"] is True


@pytest.mark.asyncio
async def test_sampling_callback_registration(basic_server):
    """Test registering a custom sampling callback"""

    @basic_server.sampling()
    async def mock_sampling(context, params):
        return types.CreateMessageResult(
            message=types.SamplingMessage(
                role="assistant", content=types.TextContent(text="Test response")
            )
        )

    # Verify the callback was registered
    assert basic_server._samplingCallback == mock_sampling


@pytest.mark.asyncio
async def test_default_sampling_callback(basic_server):
    """Test the default sampling callback behavior"""
    params = types.CreateMessageRequestParams(
        messages=[], maxTokens=100  # Add required field
    )
    context = Mock()

    result = await basic_server._samplingCallback(context, params)
    assert isinstance(result, types.ErrorData)
    assert result.code == types.INVALID_REQUEST
    assert result.message == "Sampling not supported"


@pytest.mark.asyncio
async def test_guardrail_message_check(
    server_with_guardrails, mock_guardrail, mock_request_context
):
    """Test guardrail message checking"""

    # Register a mock sampling callback
    async def mock_sampling(context, params):
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(type="text", text="Test response"),
            model="test-model",
        )

    server_with_guardrails._samplingCallback = mock_sampling

    # Create a test message
    message = types.SamplingMessage(
        role="user",
        content=types.TextContent(
            type="text", text="Test message"  # Add required field
        ),
    )

    # Create request parameters
    params = types.CreateMessageRequestParams(
        messages=[message], maxTokens=100  # Add required field
    )

    # Test successful guardrail check
    mock_guardrail.run.return_value = None
    handler = server_with_guardrails._mcp_server.request_handlers[
        types.CreateMessageRequest
    ]
    result = await handler(
        types.CreateMessageRequest(
            method="sampling/createMessage", params=params  # Add required method field
        )
    )
    assert not isinstance(result, types.ErrorData)

    # Test failed guardrail check
    mock_guardrail.run.side_effect = GuardrailException("Test guardrail violation")
    result = await handler(
        types.CreateMessageRequest(
            method="sampling/createMessage", params=params  # Add required method field
        )
    )
    assert isinstance(result, types.ErrorData)
    assert "Request blocked by guardrails" in result.message


def test_sse_app_creation(basic_server):
    """Test SSE app creation with authentication middleware"""
    app = basic_server.sse_app()
    assert app is not None
    # Verify routes are properly configured
    routes = [route for route in app.routes]
    assert len(routes) == 2  # Should have SSE path and message path routes


def test_patched_get_capabilities(basic_server):
    """Test the patched get_capabilities method"""
    # Create a proper mock for notification options with required attributes
    notification_options = Mock()
    notification_options.prompts_changed = False
    notification_options.resources_changed = False
    notification_options.tools_changed = False
    notification_options.logging_changed = False

    experimental_capabilities = {}

    capabilities = basic_server.patched_get_capabilities(
        notification_options, experimental_capabilities
    )

    assert capabilities.experimental is not None
    assert "hmcp" in capabilities.experimental
    assert capabilities.experimental["hmcp"]["sampling"] is True
    assert capabilities.experimental["hmcp"]["guardrails"] is False


@pytest.mark.asyncio
async def test_sampling_handler_with_valid_message(basic_server, mock_request_context):
    """Test sampling handler with a valid message"""

    # Create a mock sampling callback
    async def mock_sampling(context, params):
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(type="text", text="Test response"),
            model="test-model",
        )

    basic_server._samplingCallback = mock_sampling

    # Create test message and request
    message = types.SamplingMessage(
        role="user",
        content=types.TextContent(
            type="text", text="Test message"  # Add required field
        ),
    )
    params = types.CreateMessageRequestParams(
        messages=[message], maxTokens=100  # Add required field
    )
    request = types.CreateMessageRequest(
        method="sampling/createMessage", params=params  # Add required method field
    )

    # Get the handler and execute it
    handler = basic_server._mcp_server.request_handlers[types.CreateMessageRequest]
    result = await handler(request)

    assert isinstance(result, types.ServerResult)
    assert result.root.role == "assistant"
    assert result.root.content.text == "Test response"
    assert result.root.model == "test-model"


def test_server_initialization_with_custom_settings(
    mock_guardrail_class, mock_openai_api_key
):
    """Test server initialization with custom settings"""
    # Mock the Guardrail instance
    mock_guardrail_instance = Mock(spec=Guardrail)
    mock_guardrail_class.return_value = mock_guardrail_instance

    server = HMCPServer(
        name="test_server",
        host="127.0.0.1",
        port=9000,
        version="2.0.0",
        instructions="Test instructions",
        debug=True,
        log_level="DEBUG",
        enable_guardrails=True,
    )

    assert server.name == "test_server"
    assert server.settings.host == "127.0.0.1"
    assert server.settings.port == 9000
    assert server.settings.debug is True
    assert server.settings.log_level == "DEBUG"
    assert "hmcp" in server.experimentalCapabilities
    assert server.experimentalCapabilities["hmcp"]["sampling"] is True
    assert server.experimentalCapabilities["hmcp"]["version"] == "0.1.0"
    assert server.experimentalCapabilities["hmcp"]["guardrails"] is True
    mock_guardrail_class.assert_called_once()


def test_server_initialization_with_guardrail_config_path(
    tmp_path, mock_guardrail_class, mock_openai_api_key
):
    """Test server initialization with guardrail config path"""
    config_path = tmp_path / "guardrail_config.yaml"
    config_content = """
models:
  - type: main
    engine: openai
    model: gpt-4
instructions:
  - type: general
    content: Be helpful and concise.
"""
    config_path.write_text(config_content)

    # Mock the Guardrail instance
    mock_guardrail_instance = Mock(spec=Guardrail)
    mock_guardrail_class.return_value = mock_guardrail_instance

    server = HMCPServer(
        name="test_server",
        enable_guardrails=True,
        guardrail_config_path=str(config_path),
    )

    mock_guardrail_class.assert_called_once_with(config_path=str(config_path))
    assert server.enable_guardrails is True


@pytest.mark.asyncio
async def test_guardrail_check_with_empty_messages(
    server_with_guardrails, mock_guardrail, mock_request_context
):
    """Test guardrail check with empty messages list"""

    # Register a mock sampling callback
    async def mock_sampling(context, params):
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(type="text", text="Test response"),
            model="test-model",
        )

    server_with_guardrails._samplingCallback = mock_sampling

    # Create request parameters with empty messages
    params = types.CreateMessageRequestParams(messages=[], maxTokens=100)

    # Test guardrail check
    handler = server_with_guardrails._mcp_server.request_handlers[
        types.CreateMessageRequest
    ]
    result = await handler(
        types.CreateMessageRequest(method="sampling/createMessage", params=params)
    )
    assert not isinstance(result, types.ErrorData)
    mock_guardrail.run.assert_not_called()


@pytest.mark.asyncio
async def test_guardrail_check_with_list_content(
    server_with_guardrails, mock_guardrail, mock_request_context
):
    """Test guardrail check with list content"""

    # Register a mock sampling callback
    async def mock_sampling(context, params):
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(type="text", text="Test response"),
            model="test-model",
        )

    server_with_guardrails._samplingCallback = mock_sampling

    # Create a test message with concatenated content
    message = types.SamplingMessage(
        role="user", content=types.TextContent(type="text", text="Part 1\nPart 2")
    )

    params = types.CreateMessageRequestParams(messages=[message], maxTokens=100)

    request = types.CreateMessageRequest(method="sampling/createMessage", params=params)

    # Mock the guardrail's run method
    mock_guardrail.run.return_value = "Approved response"

    # Test the guardrail check through the sampling handler
    handler = server_with_guardrails._mcp_server.request_handlers[
        types.CreateMessageRequest
    ]
    result = await handler(request)

    assert isinstance(result, types.ServerResult)
    mock_guardrail.run.assert_called_once_with("Part 1\nPart 2")


@pytest.mark.asyncio
async def test_guardrail_check_with_non_guardrail_exception(
    server_with_guardrails, mock_guardrail, mock_request_context
):
    """Test guardrail check when a non-GuardrailException is raised"""

    # Register a mock sampling callback
    async def mock_sampling(context, params):
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(type="text", text="Test response"),
            model="test-model",
        )

    server_with_guardrails._samplingCallback = mock_sampling

    # Create a test message
    message = types.SamplingMessage(
        role="user", content=types.TextContent(type="text", text="Test message")
    )

    # Create request parameters
    params = types.CreateMessageRequestParams(messages=[message], maxTokens=100)

    # Test guardrail check with non-GuardrailException
    mock_guardrail.run.side_effect = ValueError("Unexpected error")
    handler = server_with_guardrails._mcp_server.request_handlers[
        types.CreateMessageRequest
    ]
    result = await handler(
        types.CreateMessageRequest(method="sampling/createMessage", params=params)
    )
    # Should continue processing
    assert not isinstance(result, types.ErrorData)


@pytest.mark.asyncio
async def test_sampling_callback_error(basic_server, mock_request_context):
    """Test handling of errors from sampling callback"""

    # Create a mock sampling callback that raises an exception
    async def mock_sampling(context, params):
        raise ValueError("Test error")

    basic_server._samplingCallback = mock_sampling

    # Create test message and request
    message = types.SamplingMessage(
        role="user", content=types.TextContent(type="text", text="Test message")
    )
    params = types.CreateMessageRequestParams(messages=[message], maxTokens=100)
    request = types.CreateMessageRequest(method="sampling/createMessage", params=params)

    # Get the handler and execute it
    handler = basic_server._mcp_server.request_handlers[types.CreateMessageRequest]
    with pytest.raises(ValueError, match="Test error"):
        await handler(request)
