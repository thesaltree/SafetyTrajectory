import os
from openai import AsyncOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

ATTACKER_MODEL = "openai/gpt-4o-mini"
JUDGE_MODEL = "openai/gpt-4o-mini"

DEFAULT_MAX_TURNS = 10
DEFAULT_TEMPERATURE_ATTACKER = 1.0
DEFAULT_TEMPERATURE_TARGET = 0.9

_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/thesaltree/SafetyTrajectory",
    "X-Title": "SafetyTrajectory Eval Framework",
}


def get_async_client() -> AsyncOpenAI:
    """Return a configured AsyncOpenAI client pointing at the OpenRouter gateway."""
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "No API key found. Set OPENROUTER_API_KEY (or OPENAI_API_KEY) in your environment."
        )

    return AsyncOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers=_OPENROUTER_HEADERS,
    )
