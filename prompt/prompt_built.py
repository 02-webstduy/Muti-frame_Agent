


def system_prompt() :
    return """
    你是一个智能体，你的任务是根据用户的问题，从记忆中提取相关信息，然后根据提取的信息，生成一个符合要求的回答。
    """
def memory_prompt(memory) :
    return f"""
    记忆：{memory}
    """
def user_prompt(query) :
    return f"""
    用户问题：{query}
    """


def build_prompt(system, memory, query) :
    return f"""
SYSTEM_PROMPT: {system}
MEMORY_PROMPT: {memory}
USER: {query}
"""
