"""LlamaIndex model factory."""
from __future__ import annotations

from typing import Any, Dict

from agents.model_config import load_model_config, optional_model_kwargs


def create_llm(config: Dict[str, Any]) -> Any:
    """Create the OpenAI-compatible LlamaIndex LLM from config."""
    from llama_index.llms.openai import OpenAI

    model = load_model_config(config)
    kwargs = optional_model_kwargs(model)
    api_base = model.base_url

    return OpenAI(
        model=model.name,
        api_key=model.api_key,
        api_base=api_base,
        **kwargs,
    )


__all__ = ["create_llm"]
