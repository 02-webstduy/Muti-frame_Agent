from __future__ import annotations

from pprint import pprint
from typing import Any
from pipeline.answer_process import answer_process
from BaseAgent import create_agent, load_config
from memory.retrieval import store_memory

def run_agent(query: str, config_path: str = "config/config.yaml") -> Any:
    config = load_config(config_path)
    agent = create_agent(config)

    final_answer = answer_process(query)

    if isinstance(agent, dict):
        return {name: current_agent.run(final_answer) for name, current_agent in agent.items()}
    return agent.run(final_answer)
#这里agent中是封装llm和各种配置的地方，然后.run只负责将message传给llm
#所以对于任何输入，或者多步模型的调用都是在其他层实现的


if __name__ == "__main__":
    query = input("请输入您的问题：")
    result = run_agent(query)
    store_memory(query, result["message"])
    print(result)
