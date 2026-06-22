from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

import yaml


def _load_langgraph_agent():
    from agents.langgraph.langgraph_agent import LangGraphAgent

    return LangGraphAgent


def _load_smolagent():
    from agents.smolagent.smolagent_agent import SmolAgent

    return SmolAgent


def _load_llama_index_agent():
    from agents.LlamaIndex.LlamaIndex_agent import LlamaIndexAgent

    return LlamaIndexAgent


AgentLoader = Callable[[], type]


def _create_backend_agent(
    backend: str,
    agent_loaders: Dict[str, AgentLoader],
    config: Dict[str, Any],
) -> Any:
    try:
        return agent_loaders[backend]()(config)
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            f"Backend '{backend}' requires missing package '{error.name}'. "
            "Install that framework dependency before selecting this backend."
        ) from error


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load the yaml config used to select the agent backend."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def create_agent(config: Dict[str, Any]) -> Any:
    """Create one agent, or all agents together, from config."""
    backend = str(config.get("agent_backend", "langgraph")).strip().lower()

    agent_loaders: Dict[str, AgentLoader] = {
        "langgraph": _load_langgraph_agent,
        "smolagent": _load_smolagent,
        "llama_index": _load_llama_index_agent,
    }

    if backend == "all":
        return {
            name: _create_backend_agent(name, agent_loaders, config)
            for name in agent_loaders
        }

    if backend not in agent_loaders:
        supported = ", ".join([*agent_loaders.keys(), "all"])
        raise ValueError(f"Unsupported agent_backend '{backend}'. Supported values: {supported}")

    return _create_backend_agent(backend, agent_loaders, config)
