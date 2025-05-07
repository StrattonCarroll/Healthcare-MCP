import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any
import mcp.types as types
from mcp import ClientSession
from hmcp.client.hmcp_client import HMCPClient


@pytest.fixture
def mock_session():
    """Mock ClientSession"""
    session = Mock(spec=ClientSession)
    session.send_request = AsyncMock()
    return session


@pytest.fixture
def hmcp_client(mock_session):
    """Create HMCPClient instance with mocked session"""
    return HMCPClient(mock_session)


@pytest.mark.asyncio
async def test_create_message_basic(hmcp_client, mock_session):
    """Test basic create_message functionality"""
    # Setup test data
    messages = [
        types.SamplingMessage(
            role="user", content=types.TextContent(type="text", text="Hello")
        ),
        types.SamplingMessage(
            role="assistant", content=types.TextContent(type="text", text="Hi there!")
        ),
    ]
    expected_result = types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text="Test response"),
        model="test-model",
        stopReason="stop",
    )
    mock_session.send_request.return_value = expected_result

    # Call the method
    result = await hmcp_client.create_message(messages)

    # Verify the result
    assert result == expected_result

    # Verify the request was sent correctly
    mock_session.send_request.assert_called_once()
    request = mock_session.send_request.call_args[0][0]
    assert request.root.method == "sampling/createMessage"
    assert request.root.params.messages == messages
    assert request.root.params.maxTokens == 1000
    assert request.root.params.temperature is None
    assert request.root.params.topP is None
    assert request.root.params.stop is None
    assert request.root.params.metadata is None


@pytest.mark.asyncio
async def test_create_message_with_all_params(hmcp_client, mock_session):
    """Test create_message with all optional parameters"""
    # Setup test data
    messages = [
        types.SamplingMessage(
            role="user", content=types.TextContent(type="text", text="Hello")
        )
    ]
    max_tokens = 500
    temperature = 0.7
    top_p = 0.9
    stop = ["\n", "Human:"]
    metadata = {"key": "value"}

    expected_result = types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text="Test response"),
        model="test-model",
        stopReason="stop",
    )
    mock_session.send_request.return_value = expected_result

    # Call the method
    result = await hmcp_client.create_message(
        messages=messages,
        maxTokens=max_tokens,
        temperature=temperature,
        topP=top_p,
        stop=stop,
        metadata=metadata,
    )

    # Verify the result
    assert result == expected_result

    # Verify the request was sent correctly
    mock_session.send_request.assert_called_once()
    request = mock_session.send_request.call_args[0][0]
    assert request.root.method == "sampling/createMessage"
    assert request.root.params.messages == messages
    assert request.root.params.maxTokens == max_tokens
    assert request.root.params.temperature == temperature
    assert request.root.params.topP == top_p
    assert request.root.params.stop == stop
    assert request.root.params.metadata == metadata


@pytest.mark.asyncio
async def test_create_message_with_empty_messages(hmcp_client, mock_session):
    """Test create_message with empty message list"""
    messages: List[types.SamplingMessage] = []
    expected_result = types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text="Test response"),
        model="test-model",
        stopReason="stop",
    )
    mock_session.send_request.return_value = expected_result

    result = await hmcp_client.create_message(messages)

    assert result == expected_result
    mock_session.send_request.assert_called_once()
    request = mock_session.send_request.call_args[0][0]
    assert request.root.params.messages == messages


@pytest.mark.asyncio
async def test_create_message_with_error_response(hmcp_client, mock_session):
    """Test create_message with error response from server"""
    messages = [
        types.SamplingMessage(
            role="user", content=types.TextContent(type="text", text="Hello")
        )
    ]
    error_response = types.ErrorData(code=400, message="Invalid request")
    mock_session.send_request.return_value = error_response

    result = await hmcp_client.create_message(messages)

    assert result == error_response
    mock_session.send_request.assert_called_once()


@pytest.mark.asyncio
async def test_create_message_with_long_messages(hmcp_client, mock_session):
    """Test create_message with long message content"""
    messages = [
        types.SamplingMessage(
            role="user",
            content=types.TextContent(
                type="text", text="This is a very long message " * 100
            ),
        )
    ]
    expected_result = types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text="Test response"),
        model="test-model",
        stopReason="stop",
    )
    mock_session.send_request.return_value = expected_result

    result = await hmcp_client.create_message(messages)

    assert result == expected_result
    mock_session.send_request.assert_called_once()
    request = mock_session.send_request.call_args[0][0]
    assert request.root.params.messages == messages


@pytest.mark.asyncio
async def test_create_message_with_special_characters(hmcp_client, mock_session):
    """Test create_message with special characters in content"""
    messages = [
        types.SamplingMessage(
            role="user",
            content=types.TextContent(
                type="text", text="Special chars: !@#$%^&*()_+{}|:\"<>?[]\\;',./"
            ),
        )
    ]
    expected_result = types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text="Test response"),
        model="test-model",
        stopReason="stop",
    )
    mock_session.send_request.return_value = expected_result

    result = await hmcp_client.create_message(messages)

    assert result == expected_result
    mock_session.send_request.assert_called_once()
    request = mock_session.send_request.call_args[0][0]
    assert request.root.params.messages == messages


@pytest.mark.asyncio
async def test_create_message_with_metadata_types(hmcp_client, mock_session):
    """Test create_message with different metadata value types"""
    messages = [
        types.SamplingMessage(
            role="user", content=types.TextContent(type="text", text="Hello")
        )
    ]
    metadata = {
        "string": "value",
        "number": 123,
        "boolean": True,
        "null": None,
        "array": [1, 2, 3],
        "object": {"nested": "value"},
    }
    expected_result = types.CreateMessageResult(
        role="assistant",
        content=types.TextContent(type="text", text="Test response"),
        model="test-model",
        stopReason="stop",
    )
    mock_session.send_request.return_value = expected_result

    result = await hmcp_client.create_message(messages, metadata=metadata)

    assert result == expected_result
    mock_session.send_request.assert_called_once()
    request = mock_session.send_request.call_args[0][0]
    assert request.root.params.metadata == metadata
