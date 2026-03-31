from __future__ import annotations

import time

from openai import APIError, AzureOpenAI, RateLimitError

from app.core.config import get_settings


class AzureOpenAIClient:
    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )

    def _extract_retry_after(self, error: Exception, attempt: int) -> float:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {}) or {}
        retry_after = headers.get("retry-after") or headers.get("x-ratelimit-reset-requests")
        if retry_after:
            try:
                return max(float(retry_after), self.settings.azure_openai_retry_base_seconds)
            except (TypeError, ValueError):
                pass
        return self.settings.azure_openai_retry_base_seconds * attempt

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        batch_size = max(1, self.settings.azure_openai_embedding_batch_size)

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            for attempt in range(1, self.settings.azure_openai_max_retries + 1):
                try:
                    response = self.client.embeddings.create(
                        model=self.settings.azure_openai_embedding_deployment,
                        input=batch,
                    )
                    vectors.extend(item.embedding for item in response.data)
                    break
                except (RateLimitError, APIError) as exc:
                    if attempt >= self.settings.azure_openai_max_retries:
                        raise
                    time.sleep(self._extract_retry_after(exc, attempt))
        return vectors

    def chat_completion(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        for attempt in range(1, self.settings.azure_openai_max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.settings.azure_openai_chat_deployment,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return response.choices[0].message.content or ""
            except (RateLimitError, APIError) as exc:
                if attempt >= self.settings.azure_openai_max_retries:
                    raise
                time.sleep(self._extract_retry_after(exc, attempt))
        return ""
