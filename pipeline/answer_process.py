from prompt.prompt_built import build_prompt, system_prompt, memory_prompt, user_prompt
from memory.retrieval import format_memory_context, retrieve_memory

def answer_process(query: str):
    #最后的处理，这里将直接把最终的输入返回给llm
    memories = retrieve_memory(query)
    memory_context = format_memory_context(memories)
    built_prompt = build_prompt(system_prompt(), memory_prompt(memory_context), user_prompt(query))
    return built_prompt
