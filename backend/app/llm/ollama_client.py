# ──────────────────────────────────────────────
# Ollama LLM Client (Local Inference)
# Uses OpenAI-compatible /v1/chat/completions API
# Supports multi-model: different models for RAG vs meetings
# ──────────────────────────────────────────────
import httpx
from typing import Optional
from app.config import OLLAMA_BASE_URL, OLLAMA_RAG_MODEL, OLLAMA_MEETING_MODEL


class OllamaClient:
    """
    Client for communicating with a local Ollama server.

    Ollama runs local LLMs and exposes an OpenAI-compatible API.
    Supports per-request model override for multi-model workflows.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_RAG_MODEL,
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
        model: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Generate a response from the local LLM.

        Args:
            prompt: The full prompt (system + context + question).
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Maximum number of tokens in the response.
            model: Override model for this request (e.g. mistral for meetings).
            top_k: Restrict sampling pool for faster generation.

        Returns:
            The generated text response.
        """
        use_model = model or self.model

        payload = {
            "model": use_model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        # Add optional Ollama-specific options
        options = {}
        if top_k is not None:
            options["top_k"] = top_k
        if max_tokens:
            options["num_predict"] = max_tokens
        if options:
            payload["options"] = options

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

    async def is_available(self, model: Optional[str] = None) -> bool:
        """Check if Ollama server is running and a model is loaded."""
        check_model = model or self.model
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    return any(
                        check_model in name for name in model_names
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
