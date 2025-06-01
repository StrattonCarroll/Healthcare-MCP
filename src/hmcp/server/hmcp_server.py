from __future__ import annotations
import logging
from typing import Any, Literal, Protocol, TypeVar, List, Sequence, Optional
import uvicorn
import mcp.types as types
from mcp.server.fastmcp import FastMCP
from mcp.shared.context import RequestContext
from mcp.server.lowlevel.server import NotificationOptions
from mcp.server.sse import SseServerTransport
from mcp.server.stdio import stdio_server
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.requests import Request
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send
from starlette.middleware import Middleware
from mcp.server.models import InitializationOptions
from pydantic import RootModel
from hmcp.shared.auth import OAuthServer, AuthConfig, jwt_handler
from hmcp.server.fastmcp_auth import AuthMiddleware
from hmcp.shared.guardrail_config.guardrail import Guardrail, GuardrailException

# Configure logging for the HMCP server module
logger = logging.getLogger(__name__)

# Extended server result type to include CreateMessageResult
# This dynamically extends the MCP ServerResult type to support HMCP's extended functionality
new_server_result_base = RootModel[
    types.EmptyResult
    | types.InitializeResult
    | types.CompleteResult
    | types.GetPromptResult
    | types.ListPromptsResult
    | types.ListResourcesResult
    | types.ListResourceTemplatesResult
    | types.ReadResourceResult
    | types.CallToolResult
    | types.ListToolsResult
    | types.CreateMessageResult
]

# Dynamically create a new class that extends the base ServerResult with our additions
types.ServerResult = type("ServerResult", (new_server_result_base,), {})

types.StopReason = Literal["endTurn", "stopSequence", "maxTokens", "infoNeeded"] | str


class SamplingFnT(Protocol):
    """
    Protocol defining the signature for sampling callback functions.

    Sampling callbacks must accept a context and parameters, and return either
    a message result or an error. This protocol ensures type safety when implementing
    custom sampling handlers.

    Args:
        context: The request context containing metadata about the client request
        params: Parameters for the sampling operation including messages and generation settings

    Returns:
        Either a successful CreateMessageResult or an ErrorData object
    """

    async def __call__(
        self,
        context: RequestContext[Any, Any],
        params: types.CreateMessageRequestParams,
    ) -> types.CreateMessageResult | types.ErrorData: ...


async def _default_sampling_callback(
    context: RequestContext[Any, Any],
    params: types.CreateMessageRequestParams,
) -> types.CreateMessageResult | types.ErrorData:
    """
    Default sampling callback that returns an error indicating sampling is not supported.

    This is used when no custom sampling callback is provided to the server.
    It ensures that clients receive a meaningful error message rather than
    a method-not-implemented error.

    Args:
        context: The request context (unused in the default implementation)
        params: The sampling parameters (unused in the default implementation)

    Returns:
        An ErrorData object with an appropriate error message
    """
    return types.ErrorData(
        code=types.INVALID_REQUEST,
        message="Sampling not supported",
    )


class HMCPServer(FastMCP):
    """
    HMCP Server extends the FastMCP Server with sampling capability.

    This class provides an implementation of the HMCP (Healthcare Model Context Protocol)
    server that adds text generation capabilities on top of the standard MCP protocol.
    HMCP servers can handle CreateMessageRequest messages from clients and generate
    text completions based on provided prompts and context.

    The server advertises its sampling capabilities to clients during initialization
    and provides a flexible API for implementing custom sampling logic through
    callback functions.
    """

    def __init__(
        self,
        name: str,
        host: str = "0.0.0.0",
        port: int = 8050,
        debug: bool = False,
        log_level: str = "INFO",
        auth_config: Optional[AuthConfig] = None,
        version: str | None = None,
        instructions: str | None = None,
        samplingCallback: SamplingFnT | None = None,
        enable_guardrails: bool = True,
        guardrail_config_path: Optional[str] = None,
        guardrail_instance: Optional[Guardrail] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the HMCP Server.

        Args:
            name: The name of the server that will be reported to clients
            version: The version of the server (optional), used for compatibility checks
            instructions: Human-readable instructions for using the server (optional)
            samplingCallback: A callback function to handle sampling requests (optional)
                              If not provided, the server will return errors for sampling requests
            enable_guardrails: Whether to enable guardrail checks for incoming messages (default: True)
            guardrail_config_path: Custom path to guardrail configuration (default: None, uses standard path)
            guardrail_instance: Pre-configured Guardrail instance (default: None, creates a new instance)
            **kwargs: Additional settings to pass to the underlying FastMCP implementation
                       These can include configuration for logging, transports, etc.
        """
        # Initialize the parent FastMCP server with standard settings
        super().__init__(
            name=name,
            host=host,
            port=port,
            instructions=instructions,
            debug=debug,
            log_level=log_level,
            *args,
            **kwargs,
        )

        # Initialize auth components if not provided
        self.auth_config = auth_config or AuthConfig()
        self.oauth_server = OAuthServer(self.auth_config)
        self.jwt_handler = jwt_handler.JWTHandler(self.auth_config)

        # Initialize guardrails
        self.enable_guardrails = enable_guardrails
        if enable_guardrails:
            if guardrail_instance:
                self.guardrail = guardrail_instance
            elif guardrail_config_path:
                self.guardrail = Guardrail(config_path=guardrail_config_path)
            else:
                self.guardrail = Guardrail()
            logger.info(f"Guardrails enabled for {name} server")
        else:
            self.guardrail = None
            logger.info(f"Guardrails disabled for {name} server")

        # Define experimental capabilities with sampling for advertisement to clients
        # This allows clients to detect that this server supports HMCP sampling features
        self.experimentalCapabilities = {
            "hmcp": {
                "sampling": True,
                "version": "0.1.0",
                "guardrails": enable_guardrails,
            }
        }

        # Store the sampling callback or use the default one if none provided
        # The callback will be invoked whenever a CreateMessageRequest is received
        self._samplingCallback = samplingCallback or _default_sampling_callback

        # Register the handler for processing CreateMessageRequest messages
        self._registerSamplingHandler()

        # Override the get_capabilities method to include our custom capabilities
        # This is necessary to add sampling capabilities to the server's advertised features
        self._original_get_capabilities = self._mcp_server.get_capabilities
        self._mcp_server.get_capabilities = self.patched_get_capabilities

    def sse_app(self, mount_path: str | None = None) -> Starlette:
        """Return an instance of the SSE server app with authentication middleware."""
        logger.info("Creating SSE app with authentication middleware")
        sse = SseServerTransport(self.settings.message_path)

        async def handle_sse(scope: Scope, receive: Receive, send: Send):
            # Add client ID from auth context into request context if available

            async with sse.connect_sse(
                scope,
                receive,
                send,
            ) as streams:
                await self._mcp_server.run(
                    streams[0],
                    streams[1],
                    self._mcp_server.create_initialization_options(),
                )
            return Response()

        async def sse_endpoint(request: Request) -> Response:
            # Convert the Starlette request to ASGI parameters
            return await handle_sse(request.scope, request.receive, request._send)

        # Create routes
        routes: list[Route | Mount] = [
            Route(self.settings.sse_path, endpoint=sse_endpoint),
            Mount(self.settings.message_path, app=sse.handle_post_message),
        ]

        # Add authentication middleware
        middleware: list[Middleware] = [
            Middleware(
                AuthMiddleware,
                auth_config=self.auth_config,
            ),
        ]

        # Create the base app
        app = Starlette(
            debug=self.settings.debug,
            routes=routes,
            middleware=middleware,
        )

        return app

    def _registerSamplingHandler(self):
        """
        Register the handler for CreateMessageRequest.

        This method sets up the request handler that will be called when
        a client sends a CreateMessageRequest to the server. The handler
        delegates the request processing to the registered sampling callback
        and properly formats the response.

        This is an internal method that is called during server initialization.
        """

        async def samplingHandler(req: types.CreateMessageRequest):
            # Get the current request context from the MCP server
            # This contains information about the client and the request
            ctx = self._mcp_server.request_context

            # Extract message content for guardrail checks
            if self.enable_guardrails and self.guardrail:
                latest_message = (
                    req.params.messages[-1] if req.params.messages else None
                )
                if latest_message:
                    message_content = ""
                    if isinstance(latest_message.content, list):
                        message_content = "".join(
                            [
                                content.text
                                for content in latest_message.content
                                if isinstance(content, types.TextContent)
                            ]
                        )
                    elif isinstance(latest_message.content, types.TextContent):
                        message_content = latest_message.content.text

                    logger.debug(
                        f"Running guardrail check on message: {message_content[:100]}..."
                    )

                    # Apply guardrail checks
                    try:
                        await self.guardrail.run(message_content)
                    except GuardrailException as e:
                        logger.warning(f"Guardrail blocked message: {str(e)}")
                        return types.ErrorData(
                            code=types.INVALID_REQUEST,
                            message=f"Request blocked by guardrails: {str(e)}",
                        )
                    except Exception as e:
                        logger.error(f"Error in guardrail check: {str(e)}")
                        # Continue processing if guardrail fails for any reason

            # Process the request using the registered sampling callback
            # The callback is responsible for generating text based on the provided messages
            response = await self._samplingCallback(ctx, req.params)

            # Return the response directly if it's an error, otherwise wrap it in a ServerResult
            # This ensures proper type handling in the MCP protocol
            if isinstance(response, types.ErrorData):
                return response
            else:
                return types.ServerResult(response)

        # Register our handler to process CreateMessageRequest messages
        # This makes the server respond to the "sampling/createMessage" method
        self._mcp_server.request_handlers[types.CreateMessageRequest] = samplingHandler

    def sampling(self):
        """
        Decorator to register a sampling callback function.

        This provides a convenient way to define the sampling logic for the server.
        The decorated function will be called whenever the server receives a
        CreateMessageRequest.

        Returns:
            A decorator function that registers the decorated function as the sampling callback

        Example:
            @hmcp_server.sampling()
            async def handle_sampling(context, params):
                # Generate a response based on the messages in params.messages
                return types.CreateMessageResult(
                    message=types.SamplingMessage(
                        role="assistant",
                        content="Generated response here"
                    )
                )
        """

        def decorator(func: SamplingFnT):
            logger.debug("Registering sampling handler")
            self._samplingCallback = func
            return func

        return decorator

    def patched_get_capabilities(
        self,
        notification_options: NotificationOptions,
        experimental_capabilities: dict[str, dict[str, Any]],
    ) -> types.ServerCapabilities:
        """
        Override the get_capabilities method to provide custom HMCP capabilities.

        This method extends the standard MCP capabilities with HMCP-specific
        capabilities, allowing clients to discover the server's sampling features.

        Args:
            notification_options: Options for configuring server notifications
            experimental_capabilities: Additional experimental capabilities to include

        Returns:
            ServerCapabilities object with HMCP sampling capabilities included
        """
        # Call the parent class method to get standard capabilities
        capabilities = self._original_get_capabilities(
            notification_options, experimental_capabilities
        )

        # Add custom HMCP-specific capabilities to the experimental section
        # This advertises the server's sampling functionality to clients
        capabilities.experimental = {
            **capabilities.experimental,
            **self.experimentalCapabilities,
        }

        logger.debug("Custom HMCP capabilities added: %s", capabilities.experimental)

        return capabilities
