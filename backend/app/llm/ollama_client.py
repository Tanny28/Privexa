# ──────────────────────────────────────────────
# Ollama LLM Client (Local Inference)
# Uses OpenAI-compatible /v1/chat/completions API
# ──────────────────────────────────────────────
import httpx
from typing import Optional
from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL


class OllamaClient:
    """
    Client for communicating with a local Ollama server.

    Ollama runs Llama 3 (or other models) locally and exposes
    an OpenAI-compatible API at http://localhost:11434/v1/
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate a response from the local LLM using the
        OpenAI-compatible chat completions endpoint.

        Args:
            prompt: The full prompt (system + context + question).
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Maximum number of tokens in the response.

        Returns:
            The generated text response.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
            return ""

    async def is_available(self) -> bool:
        """Check if Ollama server is running and the model is loaded."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    return any(
                        self.model in name for name in model_names
                    )
            return False
        except Exception:
            return False

    async def list_models(self) -> list:
        """List all available models on the Ollama server."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                return response.json().get("models", [])
        except Exception:
            return []


# Singleton instance
_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Get or create the Ollama client singleton."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client
