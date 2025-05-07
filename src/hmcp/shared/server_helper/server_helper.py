#!/usr/bin/env python3
"""
HMCP Server Helper: A utility class to simplify interactions with HMCP servers

This module provides a helper class that makes it easier to interact with HMCP servers
by abstracting away the low-level details of establishing connections, authentication,
and managing server sessions. It provides simple methods to call tools, access resources,
and use the sampling capabilities of HMCP servers.

Usage:
    from hmcp_server_helper import HMCPServerHelper

    # Create a helper for an existing HMCP server
    helper = HMCPServerHelper(host="localhost", port=8050)

    # Connect to the server
    await helper.connect()

    # Send a message for sampling
    result = await helper.create_message("Your message here")

    # Call a tool
    tool_result = await helper.call_tool("tool_name", {"param": "value"})

    # Clean up when done
    await helper.cleanup()
"""

from __future__ import annotations
import asyncio
import logging
import json
from asyncio import Lock
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional, Union, TypeVar, Type, Tuple
import os

from hmcp.shared.auth import AuthConfig, OAuthClient, jwt_handler
from hmcp.client.hmcp_client import HMCPClient
import mcp.types as types
from mcp.client.sse import sse_client
from mcp import ClientSession
from mcp.shared.exceptions import McpError
from mcp.types import (
    CreateMessageResult,
    SamplingMessage,
    TextContent,
    ErrorData,
    Tool as MCPTool,
    CallToolResult,
    ListToolsResult,
    ListPromptsResult,
    GetPromptResult,
    ListResourcesResult,
    ReadResourceResult,
)
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.types import JSONRPCMessage

# Configure logging
logger = logging.getLogger(__name__)


class HMCPServerHelper:
    """
    A helper class that simplifies interactions with HMCP servers.

    This class provides convenient methods to connect to an HMCP server,
    call its tools, access its resources, and use its sampling capabilities.
    It handles authentication and connection management automatically.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8050,
        auth_config: Optional[AuthConfig] = None,
        client_id: str = "test-client",
        client_secret: str = "test-secret",
        debug: bool = False,
    ):
        """
        Initialize the HMCP Server Helper.

        Args:
            host: The hostname or IP address of the HMCP server
            port: The port number the HMCP server is listening on
            auth_config: Optional authentication configuration
            client_id: Client ID to use for authentication
            client_secret: Client secret to use for authentication
            debug: Whether to enable debug logging
        """
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}"
        self.debug = debug

        # Set up logging
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(level=log_level)

        # Initialize auth components
        self.auth_config = auth_config or AuthConfig()
        self.jwt_handler = jwt_handler.JWTHandler(self.auth_config)
        self.oauth_client = OAuthClient(
            client_id=client_id, client_secret=client_secret, config=self.auth_config
        )

        # Initialize connection components
        self.session = None
        self.client = None
        self.exit_stack = AsyncExitStack()
        self._cleanup_lock = Lock()
        self.connected = False
        self.server_info = None

    async def connect(self) -> Dict[str, Any]:
        """
        Connect to the HMCP server.

        This method establishes a connection to the HMCP server using the SSE transport,
        initializes the client session, and returns the server information.

        Returns:
            Dict containing the server information including name, version, etc.

        Raises:
            Exception: If the connection fails
        """
        if self.connected:
            logger.warning("Already connected to server")
            return self._get_server_info()

        try:
            # Generate a JWT token
            token = self.jwt_handler.generate_token(
                client_id=self.oauth_client.client_id,
                scope=" ".join(self.auth_config.OAUTH_SCOPES[:3]),
            )

            # Set the token in the OAuth client
            self.oauth_client.set_token({"access_token": token})
            auth_headers = self.oauth_client.get_auth_header()

            logger.debug(f"Connecting to HMCP server at {self.url}/sse")

            # Setup streams and session using AsyncExitStack to properly manage cleanup
            transport = await self.exit_stack.enter_async_context(
                self._create_streams(auth_headers)
            )
            read_stream, write_stream = transport

            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            # Initialize the session
            init_result = await self.session.initialize()

            # Store server info
            self.server_info = {
                "name": init_result.serverInfo.name,
                "version": init_result.serverInfo.version,
                "protocolVersion": init_result.protocolVersion,
                "capabilities": init_result.capabilities,
            }

            # Create the HMCP client
            self.client = HMCPClient(self.session)
            self.connected = True

            logger.info(
                f"Connected to {init_result.serverInfo.name} v{init_result.serverInfo.version}"
            )
            return self.server_info

        except Exception as e:
            logger.error(f"Failed to connect to HMCP server: {str(e)}")
            # Ensure proper cleanup on failure
            await self.cleanup()
            raise

    def _create_streams(self, headers: Dict[str, str]):
        """
        Create the SSE streams to connect to the server.
        This is a helper method for connect().

        Returns:
            An async context manager that yields a tuple of (read_stream, write_stream)
        """
        return sse_client(
            url=f"{self.url}/sse",
            headers=headers,
            timeout=5,  # HTTP timeout
            sse_read_timeout=300,  # SSE read timeout (5 minutes)
        )

    async def cleanup(self) -> None:
        """
        Clean up the connection to the HMCP server.

        This method closes any open connections and frees resources.
        It should be called when the helper is no longer needed.
        """
        if not self.connected:
            return

        async with self._cleanup_lock:
            try:
                self.connected = False
                self.session = None
                self.client = None

                # Close the exit stack to clean up all context managers
                await self.exit_stack.aclose()
                logger.debug("Cleaned up HMCP server connection")

            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")

    def _ensure_connected(self) -> None:
        """
        Ensure that the helper is connected to the server.

        Raises:
            RuntimeError: If not connected to the server
        """
        if not self.connected or not self.client or not self.session:
            raise RuntimeError("Not connected to HMCP server. Call connect() first.")

    def _get_server_info(self) -> Dict[str, Any]:
        """
        Get basic server information.

        Returns:
            Dict containing server information
        """
        self._ensure_connected()
        if self.server_info:
            return self.server_info

        return {
            "name": "Unknown",
            "version": "Unknown",
            "capabilities": {},
        }

    async def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all available tools on the server.

        Returns:
            List of tools with their names, descriptions, and schemas

        Raises:
            RuntimeError: If not connected to the server
        """
        self._ensure_connected()

        try:
            result: ListToolsResult = await self.session.list_tools()

            # Convert to a more user-friendly format
            tools_list = []
            for tool in result.tools:
                tool_dict = {
                    "name": tool.name,
                    "description": tool.description,
                }

                # Add schema information if available
                if hasattr(tool, "schema") and tool.schema:
                    tool_dict["schema"] = tool.schema

                tools_list.append(tool_dict)

            return tools_list

        except Exception as e:
            logger.error(f"Error listing tools: {str(e)}")
            raise

    async def call_tool(
        self, tool_name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Call a tool on the HMCP server.

        Args:
            tool_name: The name of the tool to call
            arguments: Optional arguments to pass to the tool

        Returns:
            The result of the tool call

        Raises:
            RuntimeError: If not connected to the server
            ValueError: If the tool call returns an error
        """
        self._ensure_connected()

        try:
            result: CallToolResult = await self.session.call_tool(tool_name, arguments)

            # Convert to a dictionary for easier use
            if hasattr(result, "result") and result.result is not None:
                return result.result
            else:
                return {}

        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {str(e)}")
            raise

    async def list_resources(self) -> List[Dict[str, Any]]:
        """
        List all available resources on the server.

        Returns:
            List of resources

        Raises:
            RuntimeError: If not connected to the server
        """
        self._ensure_connected()

        try:
            result: ListResourcesResult = await self.session.list_resources()

            # Convert to a more user-friendly format
            resources_list = []
            for resource in result.resources:
                resource_dict = {
                    "uri": resource.uri,
                    "title": getattr(resource, "title", None),
                    "description": getattr(resource, "description", None),
                }
                resources_list.append(resource_dict)

            return resources_list

        except Exception as e:
            logger.error(f"Error listing resources: {str(e)}")
            raise

    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """
        Read a resource from the server.

        Args:
            uri: The URI of the resource to read

        Returns:
            The content of the resource

        Raises:
            RuntimeError: If not connected to the server
            ValueError: If the resource cannot be read
        """
        self._ensure_connected()

        try:
            result: ReadResourceResult = await self.session.read_resource(uri)

            # Convert to a dictionary for easier use
            return {
                "content": result.content,
                "contentType": getattr(result, "contentType", None),
            }

        except Exception as e:
            logger.error(f"Error reading resource {uri}: {str(e)}")
            raise

    async def list_prompts(self) -> List[Dict[str, Any]]:
        """
        List all available prompts on the server.

        Returns:
            List of prompts

        Raises:
            RuntimeError: If not connected to the server
        """
        self._ensure_connected()

        try:
            result: ListPromptsResult = await self.session.list_prompts()

            # Convert to a more user-friendly format
            prompts_list = []
            for prompt in result.prompts:
                prompt_dict = {
                    "name": prompt.name,
                    "description": getattr(prompt, "description", None),
                }
                prompts_list.append(prompt_dict)

            return prompts_list

        except Exception as e:
            logger.error(f"Error listing prompts: {str(e)}")
            raise

    async def get_prompt(
        self, name: str, arguments: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Get a prompt from the server.

        Args:
            name: The name of the prompt
            arguments: Optional arguments to pass to the prompt

        Returns:
            The prompt content

        Raises:
            RuntimeError: If not connected to the server
            ValueError: If the prompt cannot be retrieved
        """
        self._ensure_connected()

        try:
            result: GetPromptResult = await self.session.get_prompt(name, arguments)

            return result.prompt

        except Exception as e:
            logger.error(f"Error getting prompt {name}: {str(e)}")
            raise

    async def create_message(
        self,
        message: str,
        role: str = "user",
        messages_history: Optional[List[Dict[str, Any]]] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a message using the server's sampling capability.

        Args:
            message: The message content
            role: The role of the message sender (default: "user")
            messages_history: Optional history of previous messages
            model_params: Optional parameters for the model

        Returns:
            The response from the server containing role, content, model, stopReason

        Raises:
            RuntimeError: If not connected to the server
            ValueError: If message creation fails
        """
        self._ensure_connected()

        try:
            # Convert message to the format expected by HMCP
            text_content = TextContent(type="text", text=message)

            # Prepare message history
            all_messages = []

            if messages_history:
                for msg in messages_history:
                    if isinstance(msg, dict) and "content" in msg and "role" in msg:
                        if isinstance(msg["content"], str):
                            all_messages.append(
                                SamplingMessage(
                                    role=msg["role"],
                                    content=TextContent(
                                        type="text", text=msg["content"]
                                    ),
                                )
                            )
                        elif isinstance(msg["content"], TextContent):
                            all_messages.append(
                                SamplingMessage(
                                    role=msg["role"], content=msg["content"]
                                )
                            )

            # Add the current message
            if message:
                all_messages.append(SamplingMessage(role=role, content=text_content))

            # Use the client to create a message
            result = await self.client.create_message(messages=all_messages)

            # Handle error response
            if isinstance(result, ErrorData):
                error_msg = (
                    f"Error creating message: {result.message} (code: {result.code})"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Extract and return the result
            content = None
            if hasattr(result, "content"):
                content = getattr(result.content, "text", result.content)

            response = {
                "role": getattr(result, "role", "assistant"),
                "content": content,
                "model": getattr(result, "model", None),
                "stopReason": getattr(result, "stopReason", None),
            }

            return response

        except Exception as e:
            if not isinstance(e, ValueError):
                logger.error(f"Error creating message: {str(e)}")
            raise

    async def __aenter__(self) -> "HMCPServerHelper":
        """
        Enter the async context manager.

        Returns:
            The helper instance
        """
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the async context manager.
        """
        await self.cleanup()
