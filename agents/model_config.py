from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class ModelConfig:
    name: str
    api_key: str
    base_url: str
    temperature: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None
    extra: Dict[str, Any] = field(default_factory=dict)


def load_model_config(config: Dict[str, Any]) -> ModelConfig:
    """Read and validate the shared model config."""
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("config.model must be a mapping.")

    missing = [
        key
        for key in ("name", "api_key", "base_url")
        if model.get(key) is None or str(model.get(key)).strip() == ""
    ]
    if missing:
        raise ValueError(f"config.model is missing required keys: {', '.join(missing)}")

    extra = model.get("kwargs", {})
    if extra is None:
        extra = {}
    if not isinstance(extra, dict):
        raise ValueError("config.model.kwargs must be a mapping when provided.")

    return ModelConfig(
        name=str(model["name"]),
        api_key=str(model["api_key"]),
        base_url=str(model["base_url"]),
        temperature=model.get("temperature"),
        max_tokens=model.get("max_tokens"),
        timeout=model.get("timeout"),
        extra=extra,
    )


def optional_model_kwargs(model: ModelConfig) -> Dict[str, Any]:
    """Return optional parameters supported by most OpenAI-compatible clients."""
    kwargs = dict(model.extra)

    if model.temperature is not None:
        kwargs["temperature"] = model.temperature
    if model.max_tokens is not None:
        kwargs["max_tokens"] = model.max_tokens
    if model.timeout is not None:
        kwargs["timeout"] = model.timeout

    return kwargs


def extract_response_text(response: Any) -> str:
    """Normalize response objects from different agent frameworks into text."""
    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content is not None:
            return str(content)

    for attr_name in ("content", "text", "response"):
        value = getattr(response, attr_name, None)
        if value is not None:
            return str(value)

    return str(response)
