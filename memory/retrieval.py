from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List


MemoryItem = Dict[str, str]

MEMORY_FILE = Path(__file__).resolve().parent / "memory.json"
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "what",
    "when",
    "where",
    "who",
    "why",
}


def load_memory(file_path: Path | str = MEMORY_FILE) -> List[MemoryItem]:
    """Load all saved conversation memories."""
    path = Path(file_path)
    if not path.exists():
        return []

    try:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return []
        data = json.loads(content)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    memories: List[MemoryItem] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        query = str(item.get("query", "")).strip()
        response = str(item.get("response", "")).strip()
        if query or response:
            memories.append({"query": query, "response": response})

    return memories


def save_memory(
    memory_list: List[MemoryItem],
    file_path: Path | str = MEMORY_FILE,
) -> None:
    """Save all conversation memories."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(memory_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def store_memory(
    query: str,
    response: Any,
    file_path: Path | str = MEMORY_FILE,
) -> None:
    """Append one user query and final agent answer to memory."""
    memories = load_memory(file_path)
    memories.append(
        {
            "query": str(query).strip(),
            "response": str(response).strip(),
        }
    )
    save_memory(memories, file_path)


def retrieve_memory(
    query: str,
    file_path: Path | str = MEMORY_FILE,
    top_k: int = 5,
) -> List[MemoryItem]:
    """Return memories that are most related to the current query."""
    memories = load_memory(file_path)
    if not memories:
        return []

    scored_memories = [
        (_score_memory(query, memory), index, memory)
        for index, memory in enumerate(memories)
    ]

    related = [
        memory
        for score, _, memory in sorted(
            scored_memories,
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        if score > 0
    ]

    if related:
        return related[:top_k]

    return memories[-top_k:]


def format_memory_context(memories: List[MemoryItem]) -> str:
    """Format retrieved memories before injecting them into the prompt."""
    if not memories:
        return "暂无相关记忆。"

    lines: List[str] = []
    for index, memory in enumerate(memories, start=1):
        lines.append(
            f"{index}. 用户曾问：{memory['query']}\n"
            f"   当时回答：{memory['response']}"
        )

    return "\n".join(lines)


def _score_memory(query: str, memory: MemoryItem) -> int:
    query_terms = _tokenize(query)
    memory_text = f"{memory.get('query', '')} {memory.get('response', '')}"
    memory_terms = _tokenize(memory_text)

    if not query_terms or not memory_terms:
        return 0

    overlap_score = len(query_terms & memory_terms)
    contains_score = 0

    normalized_query = query.strip().lower()
    normalized_memory = memory_text.lower()
    if normalized_query and normalized_query in normalized_memory:
        contains_score += 3

    for term in query_terms:
        if term in normalized_memory:
            contains_score += 1

    return overlap_score + contains_score


def _tokenize(text: str) -> set[str]:
    normalized = text.lower()
    english_words = [
        word
        for word in re.findall(r"[a-zA-Z0-9_]+", normalized)
        if word not in STOP_WORDS
    ]
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
    return set(english_words + chinese_chars)
