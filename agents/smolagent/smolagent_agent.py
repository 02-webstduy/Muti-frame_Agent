from __future__ import annotations

from typing import Any, Dict

from agents.model_config import extract_response_text
from agents.smolagent import create_llm
from smolagents import ToolCallingAgent
from memory import store


class SmolAgent:
    """Small wrapper used to unify the project agent interface."""

    def __init__(self, config: Dict[str, Any]):


        self.config = config
        self.framework_name = "smolagent"
        self.llm = create_llm(config)
        self.agent = ToolCallingAgent(
            tools=[],
            model=self.llm,
            max_steps=config.get("max_steps", 3),
        )

    def run(self, query: str) -> Dict[str, Any]:
        response = self.agent.run(query)
        store(query,response)
        return {
            "framework": self.framework_name,
            "query": query,
            "message": extract_response_text(response),
        }
