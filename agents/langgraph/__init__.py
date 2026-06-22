"""LangGraph model factory."""
from __future__ import annotations

from typing import Any, Dict

from agents.model_config import load_model_config, optional_model_kwargs


class BaseAgentModel:
    pass


def create_llm(config: Dict[str, Any]) -> Any:
    """Create the ChatOpenAI model used by the LangGraph workflow."""
    from langchain_openai import ChatOpenAI

    model = load_model_config(config)
    return ChatOpenAI(
        model=model.name,
        api_key=model.api_key,
        base_url=model.base_url,
        **optional_model_kwargs(model),
    )


__all__ = ["BaseAgentModel", "create_llm"]
