"""smolagents model factory."""
from __future__ import annotations

from typing import Any, Dict

from agents.model_config import load_model_config, optional_model_kwargs


def create_llm(config: Dict[str, Any]) -> Any:
    """Create the OpenAI-compatible smolagents model from config."""
    from smolagents import OpenAIServerModel

    model = load_model_config(config)
    return OpenAIServerModel(
        model_id=model.name,
        api_key=model.api_key,
        api_base=model.base_url,
        **optional_model_kwargs(model),
    )


__all__ = ["create_llm"]
