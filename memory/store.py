import json
import os
from typing import Dict, Any, List

MEMORY_FILE = "memory.json"


def load_memory() -> List[Dict[str, str]]:
    """从文件加载已有的记忆数据"""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
                else:
                    return []
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    return []


def save_memory(memory_list: List[Dict[str, str]]):
    """将记忆数据保存到文件"""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory_list, f, ensure_ascii=False, indent=2)


def store(query: str, response: str):
    """存储新的查询和响应到记忆文件"""
    # 加载现有记忆
    memory_list = load_memory()

    # 添加新条目
    memory_list.append({
        "query": query,
        "response": response
    })

    # 保存更新后的记忆列表
    save_memory(memory_list)


def get_all_memories() -> List[Dict[str, str]]:
    """获取所有记忆"""
    return load_memory()


def clear_memory():
    """清空记忆文件"""
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)