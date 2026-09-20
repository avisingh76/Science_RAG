"""
graph.py  —  LangGraph agent (multi-PDF support).

Changes from v1:
  - System prompt updated to mention multiple books and source filtering
  - Everything else retained (retry, recursion limit, history window, logging)
"""

import logging
import os
import re
import operator
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from agent.tools import ALL_TOOLS

load_dotenv()

logger = logging.getLogger("science_rag.graph")

# ── Env validation ─────────────────────────────────────────────────────────
_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
_OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL")

if not _OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set in .env file.")
if not _OPENROUTER_MODEL:
    raise RuntimeError("OPENROUTER_MODEL is not set in .env file.")

RECURSION_LIMIT = 10
HISTORY_WINDOW  = 10

# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful science tutor assistant for students.
You have access to multiple science textbooks through your tools.

Available tools:
- rag_search: Search across all ingested textbooks for any science concept.
              Use the optional 'source' parameter to search within a specific book.
- sql_query:  Query metadata — list available books, page counts, chunk counts.
              Tables: chunks(chunk_id, source, page, preview, text),
                      pages(id, source, page, preview),
                      sources(filename, num_chunks, num_pages, ingested_at)
- wikipedia_search: Get real-world context and additional information.

Rules:
1. If the user mentions a specific book, use the 'source' parameter in rag_search.
2. If no book is specified, search across all books.
3. Always cite the source PDF and page number when referencing content.
4. Use sql_query to list available books when the user asks what books are available.
5. Use wikipedia_search to add real-world context after finding textbook content.
6. Keep explanations simple and student-friendly.
"""


# ── Agent state ────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


# ── LLM ───────────────────────────────────────────────────────────────────
def get_llm():
    return ChatOpenAI(
        model=_OPENROUTER_MODEL,
        api_key=_OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=1000,
    ).bind_tools(ALL_TOOLS)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _call_llm(messages: list):
    return get_llm().invoke(messages)


# ── Graph nodes ────────────────────────────────────────────────────────────
def llm_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = _call_llm(messages)
    tool_calls = getattr(response, "tool_calls", [])
    if tool_calls:
        logger.info("LLM requested tool calls: %s", [tc["name"] for tc in tool_calls])
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── Build graph ────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("llm",   llm_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")
    return graph.compile()


# ── Run agent ──────────────────────────────────────────────────────────────
def run_agent(user_message: str, history: list[dict] | None = None) -> dict:
    messages: list = []

    if history:
        window = history[-HISTORY_WINDOW:]
        for msg in window:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        logger.info("Loaded %d history messages.", len(window))

    messages.append(HumanMessage(content=user_message))
    logger.info("Running agent | message_len=%d | history_msgs=%d",
                len(user_message), len(messages) - 1)

    result = build_graph().invoke(
        {"messages": messages},
        config={"recursion_limit": RECURSION_LIMIT},
    )

    final  = result["messages"][-1]
    answer = final.content if hasattr(final, "content") else str(final)

    sources: list[str] = []
    for msg in result["messages"]:
        content = getattr(msg, "content", "") or ""
        pages   = re.findall(r"Page\s+(\d+)", content)
        sources.extend(f"Page {p}" for p in pages)

    unique_sources = list(set(sources))
    logger.info("Agent finished | sources=%s", unique_sources)
    return {"answer": answer, "sources": unique_sources}


# ── Session title ──────────────────────────────────────────────────────────
def generate_session_title(message: str) -> str:
    try:
        client = OpenAI(
            api_key=_OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
        response = client.chat.completions.create(
            model=_OPENROUTER_MODEL,
            max_tokens=15,
            messages=[{
                "role":    "user",
                "content": (
                    "Give a 3-word title for this question. "
                    f"Only return the title, nothing else: {message}"
                ),
            }],
        )
        title = response.choices[0].message.content.strip()
        logger.info("Generated session title: '%s'", title)
        return title
    except Exception:
        logger.exception("Failed to generate session title.")
        return "New Chat"