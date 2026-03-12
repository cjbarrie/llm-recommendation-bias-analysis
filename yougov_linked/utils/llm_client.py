"""
OpenRouter LLM Client for YouGov Linked Pipeline

Uses the OpenRouter API (OpenAI-compatible) to access multiple model providers
through a single API key and endpoint.

Supported models (examples):
- openai/gpt-4o-mini
- anthropic/claude-sonnet-4-5
- google/gemini-2.0-flash-001
"""

import os
import time
import random
from typing import Optional


DEFAULT_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-sonnet-4-5",
    "google/gemini-2.0-flash-001",
]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    """LLM client using OpenRouter API (OpenAI-compatible interface)."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        requests_per_minute: int = 60,
        max_retries: int = 5,
        max_tokens: int = 512,
    ):
        """
        Args:
            model: OpenRouter model ID, e.g. 'openai/gpt-4o-mini'
            api_key: OpenRouter API key. Defaults to OPENROUTER_API_KEY env var.
            requests_per_minute: Rate limit (requests/min). Default 60.
            max_retries: Max retry attempts on transient errors.
            max_tokens: Max tokens in completion.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Run: pip install openai")

        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
                "or pass api_key argument."
            )

        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._min_interval = 60.0 / requests_per_minute
        self._last_request_time = 0.0
        self._call_count = 0
        self._total_tokens = 0

        self._client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
        )

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: Input text prompt.
            temperature: Sampling temperature (0-1).

        Returns:
            Generated text string.
        """
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                self._last_request_time = time.time()
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                )
                self._call_count += 1
                usage = response.usage
                if usage:
                    self._total_tokens += usage.total_tokens or 0

                return response.choices[0].message.content or ""

            except Exception as e:
                err = str(e).lower()
                is_rate_limit = "429" in err or "rate limit" in err or "too many" in err
                is_server_err = "500" in err or "502" in err or "503" in err or "timeout" in err

                if (is_rate_limit or is_server_err) and attempt < self.max_retries - 1:
                    jitter = random.uniform(0, delay * 0.2)
                    wait = delay + jitter
                    print(f"  [OpenRouter] Attempt {attempt+1} failed ({type(e).__name__}). "
                          f"Retrying in {wait:.1f}s...")
                    time.sleep(wait)
                    delay = min(delay * 2, 120.0)
                else:
                    raise

        raise RuntimeError(f"All {self.max_retries} attempts failed for model {self.model}")

    def get_stats(self) -> dict:
        """Return usage statistics."""
        return {
            "model": self.model,
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
        }


def get_openrouter_client(model: str, **kwargs) -> OpenRouterClient:
    """Factory function to create an OpenRouterClient for a given model."""
    return OpenRouterClient(model=model, **kwargs)
