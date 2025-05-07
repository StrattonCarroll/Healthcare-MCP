from nemoguardrails import LLMRails, RailsConfig
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class GuardrailException(Exception):
    pass


class Guardrail:
    def __init__(self, config_path: str = None):
        """
        Initialize the Guardrail with configuration.

        Args:
            config_path: Optional custom path to guardrail configuration.
                         If not provided, uses the default config path.
        """

        # Use custom config path if provided, otherwise use default
        if config_path:
            config_dir = Path(config_path)
            logger.info(f"Using custom guardrail config path: {config_dir}")
        else:
            # Get current file path and parent directory
            current_file = Path(__file__)
            config_dir = current_file.parent.absolute()
            logger.debug(f"Current file path: {current_file}")
            logger.info(f"Using default guardrail config path: {config_dir}")

        self.config = RailsConfig.from_path(str(config_dir))
        self.rails = LLMRails(self.config)

    async def run(self, user_input: str) -> str:
        """
        Run guardrail checks on the user input.

        Args:
            user_input: The user message to check against guardrail rules

        Returns:
            The guardrail response

        Raises:
            GuardrailException: If the message violates guardrail rules
        """
        truncated_input = (
            user_input[:50] + "..." if len(user_input) > 50 else user_input
        )
        logger.debug(f"Running guardrail check on input: {truncated_input}")

        guardrail_response = await self.rails.generate_async(
            messages=[{"role": "user", "content": user_input}]
        )
        content = guardrail_response.get("content", "")

        # Check if request was blocked by guardrails
        if "I'm sorry, I can't respond to that" in content:
            logger.warning(f"Guardrail blocked message: {truncated_input}")
            raise GuardrailException("Request blocked by guardrails")

        return content

        return guardrail_response.get("content", "")
