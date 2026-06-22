from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.langgraph import create_llm
from agents.model_config import extract_response_text


class AgentState(TypedDict):
    query: str
    steps: List[Dict[str, Any]]
    response: Dict[str, Any]


def collect_query(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    return {
        "steps": [
            {
                "node": "collect_query",
                "message": f"Received query: {query}",
            }
        ]
    }


class LangGraphAgent:
    """LangGraph wrapper with a real LLM call inside the graph."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.framework_name = "langgraph"
        self.llm = create_llm(config)
        self.graph = self.build_graph()

    def call_llm(self, state: AgentState) -> Dict[str, Any]:
        response = self.llm.invoke(state["query"])
        steps = [
            *state.get("steps", []),
            {
                "node": "call_llm",
                "message": "Called the configured LangGraph LLM.",
            },
        ]
        return {
            "steps": steps,
            "response": {
                "framework": self.framework_name,
                "query": state["query"],
                "message": extract_response_text(response),
                "steps": steps,
            },
        }

    def build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("collect_query", collect_query)
        graph.add_node("call_llm", self.call_llm)
        graph.add_edge(START, "collect_query")
        graph.add_edge("collect_query", "call_llm")
        graph.add_edge("call_llm", END)
        return graph.compile()

    def run(self, query: str) -> Dict[str, Any]:
        result = self.graph.invoke(
            {
                "query": query,
                "steps": [],
                "response": {},
            }
        )
        return result["response"]
