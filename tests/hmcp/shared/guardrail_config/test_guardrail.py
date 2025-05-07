import os
import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
from hmcp.shared.guardrail_config.guardrail import Guardrail, GuardrailException
from nemoguardrails import LLMRails, RailsConfig


@pytest.fixture
def mock_rails_config():
    """Mock RailsConfig to avoid file system dependencies"""
    with patch("hmcp.shared.guardrail_config.guardrail.RailsConfig") as mock:
        mock.from_path.return_value = Mock(spec=RailsConfig)
        yield mock


@pytest.fixture
def mock_llm_rails():
    """Mock LLMRails to avoid actual LLM calls"""
    with patch("hmcp.shared.guardrail_config.guardrail.LLMRails") as mock:
        mock_instance = Mock(spec=LLMRails)
        mock_instance.generate_async = AsyncMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_openai_api_key():
    """Mock OpenAI API key environment variable"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        yield


def test_guardrail_initialization_with_default_config(
    mock_rails_config, mock_llm_rails, mock_openai_api_key
):
    """Test Guardrail initialization with default config path"""
    with patch("hmcp.shared.guardrail_config.guardrail.Path") as mock_path:
        # Mock the file path and parent directory
        mock_file = Mock()
        mock_file.parent.absolute.return_value = Path("/mock/path")
        mock_path.return_value = mock_file

        guardrail = Guardrail()

        # Verify config path construction
        mock_rails_config.from_path.assert_called_once_with("/mock/path")

        # Verify instance attributes
        assert guardrail.config == mock_rails_config.from_path.return_value
        assert guardrail.rails == mock_llm_rails


def test_guardrail_initialization_with_custom_config(mock_rails_config, mock_llm_rails):
    """Test Guardrail initialization with custom config path"""
    custom_path = "/custom/path/to/config"
    guardrail = Guardrail(config_path=custom_path)

    # Verify custom config path usage
    mock_rails_config.from_path.assert_called_once_with(custom_path)

    # Verify instance attributes
    assert guardrail.config == mock_rails_config.from_path.return_value
    assert guardrail.rails == mock_llm_rails


@pytest.mark.asyncio
async def test_guardrail_run_successful(mock_llm_rails):
    """Test successful guardrail run with approved message"""
    guardrail = Guardrail()
    test_input = "This is a test message"
    expected_response = "Approved response"

    # Mock the LLMRails generate_async response
    mock_llm_rails.generate_async.return_value = {"content": expected_response}

    # Run the guardrail check
    result = await guardrail.run(test_input)

    # Verify the result
    assert result == expected_response
    mock_llm_rails.generate_async.assert_called_once_with(
        messages=[{"role": "user", "content": test_input}]
    )


@pytest.mark.asyncio
async def test_guardrail_run_blocked_message(mock_llm_rails):
    """Test guardrail run with blocked message"""
    guardrail = Guardrail()
    test_input = "This is a blocked message"

    # Mock the LLMRails generate_async response with blocking message
    mock_llm_rails.generate_async.return_value = {
        "content": "I'm sorry, I can't respond to that"
    }

    # Verify that GuardrailException is raised
    with pytest.raises(GuardrailException, match="Request blocked by guardrails"):
        await guardrail.run(test_input)

    # Verify the LLMRails call
    mock_llm_rails.generate_async.assert_called_once_with(
        messages=[{"role": "user", "content": test_input}]
    )


@pytest.mark.asyncio
async def test_guardrail_run_empty_response(mock_llm_rails):
    """Test guardrail run with empty response"""
    guardrail = Guardrail()
    test_input = "This is a test message"

    # Mock the LLMRails generate_async response with empty content
    mock_llm_rails.generate_async.return_value = {}

    # Run the guardrail check
    result = await guardrail.run(test_input)

    # Verify empty string is returned
    assert result == ""
    mock_llm_rails.generate_async.assert_called_once_with(
        messages=[{"role": "user", "content": test_input}]
    )


@pytest.mark.asyncio
async def test_guardrail_run_llm_error(mock_llm_rails):
    """Test guardrail run when LLM raises an error"""
    guardrail = Guardrail()
    test_input = "This is a test message"

    # Mock the LLMRails generate_async to raise an exception
    mock_llm_rails.generate_async.side_effect = Exception("LLM error")

    # Verify that the exception is propagated
    with pytest.raises(Exception, match="LLM error"):
        await guardrail.run(test_input)

    # Verify the LLMRails call
    mock_llm_rails.generate_async.assert_called_once_with(
        messages=[{"role": "user", "content": test_input}]
    )


def test_guardrail_exception():
    """Test GuardrailException creation and message"""
    error_message = "Test error message"
    exception = GuardrailException(error_message)
    assert str(exception) == error_message
