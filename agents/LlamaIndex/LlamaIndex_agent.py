from __future__ import annotations

from typing import Any, Dict

from agents.LlamaIndex import create_llm
from agents.model_config import extract_response_text


class LlamaIndexAgent:
    """Small wrapper used to unify the project agent interface."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.framework_name = "llama_index"
        self.llm = create_llm(config)

    def run(self, query: str) -> Dict[str, Any]:
        response = self.llm.complete(query)
        return {
            "framework": self.framework_name,
            "query": query,
            "message": extract_response_text(response),
        }
