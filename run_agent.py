from __future__ import annotations

from pprint import pprint
from typing import Any

from BaseAgent import create_agent, load_config


def run_agent(query: str, config_path: str = "config/config.yaml") -> Any:
    config = load_config(config_path)
    agent = create_agent(config)
    if isinstance(agent, dict):
        return {name: current_agent.run(query) for name, current_agent in agent.items()}

    return agent.run(query)


if __name__ == "__main__":
    demo_query = ""
    result = run_agent(demo_query)
    print(result)

