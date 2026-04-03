"""
single_agent_gpt.py - Base ReAct pipeline built on the Gizmo framework.

Backbone:  GPTAgent (OpenAI Responses API)
Tools:
  - LocalSearchTool + LocalVisitTool for BCP-style local-corpus runs
  - SearchTool + VisitTool for BC-style online-tool runs

Function signature and return-dict shape stay close to
GEM/inference/single_agent_qwen.py.
"""

import json
import os
import sys
import time
from typing import Dict, Optional, Union

import tiktoken


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Gizmo.agents.base_agent import LLMConfig
from Gizmo.agents.gpt_agent import GPTAgent
from Gizmo.tools.local_search_tool import LocalSearchTool
from Gizmo.tools.local_visit_tool import LocalVisitTool
from Gizmo.tools.search_tool import SearchTool
from Gizmo.tools.visit_tool import VisitTool


CorpusSearcher = Union[LocalSearchTool, SearchTool]


def _build_visit_tool(
    config,
    corpus_searcher: CorpusSearcher,
    api_key: str,
    base_url: str,
    model: str,
):
    """Build the matching visit tool for the selected search tool."""
    visit_api_key = getattr(config, "auxiliary_model_api_key", None) or api_key
    visit_base_url = getattr(config, "auxiliary_model_base_url", None) or base_url
    visit_model = getattr(config, "auxiliary_model", None) or model

    if isinstance(corpus_searcher, LocalSearchTool):
        return LocalVisitTool(
            llm_api_key=visit_api_key,
            llm_base_url=visit_base_url,
            llm_model=visit_model,
            search_tool=corpus_searcher,
        )

    if isinstance(corpus_searcher, SearchTool):
        jina_api_key = getattr(config, "jina_api_key", "") or ""
        if not jina_api_key:
            raise ValueError(
                "config.jina_api_key is required when corpus_searcher is a SearchTool."
            )
        return VisitTool(
            jina_api_key=jina_api_key,
            llm_api_key=visit_api_key,
            llm_base_url=visit_base_url,
            llm_model=visit_model,
        )

    raise TypeError(
        "corpus_searcher must be an instance of LocalSearchTool or SearchTool."
    )


def _serialize_for_token_count(item) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        content = item.get("content")
        if item.get("type") == "message" and isinstance(content, str):
            return content
        return json.dumps(item, ensure_ascii=False)
    return str(item)


def run_react_agent(
    question: str,
    answer: str = "",
    config=None,
    corpus_searcher: Optional[CorpusSearcher] = None,
    log_dir: Optional[str] = None,
    log_name: str = "trajectory",
) -> Dict:
    """Run a Gizmo/GPT ReAct agent with either local or online search tools."""
    if corpus_searcher is None:
        raise ValueError(
            "corpus_searcher (a LocalSearchTool or SearchTool instance) is required "
            "for this pipeline."
        )

    process_start_time = time.time()

    gen_cfg: dict = dict(getattr(config, "llm_generate_cfg", {}) or {})
    default_max_output_tokens = 8192 if isinstance(corpus_searcher, SearchTool) else None

    llm_config = LLMConfig(
        max_output_tokens=gen_cfg.get(
            "max_output_tokens",
            gen_cfg.get("max_tokens", getattr(config, "max_output_tokens", default_max_output_tokens)),
        ),
        temperature=gen_cfg.get("temperature", getattr(config, "temperature", None)),
        top_p=gen_cfg.get("top_p", getattr(config, "top_p", None)),
        seed=gen_cfg.get("seed", getattr(config, "seed", None)),
        timeout=float(getattr(config, "llm_timeout", 120.0)),
        parallel_tool_calls=bool(getattr(config, "parallel_tool_calls", False)),
        reasoning_effort=getattr(config, "reasoning_effort", None),
        reasoning_summary=getattr(config, "reasoning_summary", None),
        text_verbosity=getattr(config, "text_verbosity", None),
    )

    api_key = getattr(config, "reasoning_model_api_key", "")
    base_url = getattr(config, "reasoning_model_base_url", "")
    model = getattr(config, "reasoning_model", "")

    visit_tool = _build_visit_tool(
        config=config,
        corpus_searcher=corpus_searcher,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )

    agent = GPTAgent(
        model=model,
        api_key=api_key,
        base_url=base_url,
        tools=[corpus_searcher, visit_tool],
        max_steps=getattr(config, "max_llm_call_per_run", 200),
        max_time_seconds=getattr(config, "max_time_seconds", None),
        llm_config=llm_config,
    )

    _enc = tiktoken.get_encoding("cl100k_base")
    token_limit = getattr(config, "max_tokens", 105 * 1024)

    def _count_tokens(messages: list[dict]) -> int:
        total = 0
        for item in messages:
            total += len(_enc.encode(_serialize_for_token_count(item)))
        return total

    def _on_should_stop(state):
        total = _count_tokens(agent.messages)
        if total >= token_limit:
            print(
                f"[TokenBudget] {total} tokens >= limit {token_limit}, forcing finalization."
            )
            return "token_budget"
        return None

    agent.on("should_stop", _on_should_stop)

    agent.run(question)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        agent.save_trajectory(os.path.join(log_dir, f"{log_name}.json"))

    prediction = "No answer found."
    termination = "no answer found"

    if agent.trajectory:
        last_step = agent.trajectory[-1]
        if last_step.final_content:
            prediction = last_step.final_content
            termination = "answer (final assistant response)"

    final_messages = [
        {"type": "message", "role": "developer", "content": agent.system_prompt},
        *agent.messages,
    ]
    total_process_time = time.time() - process_start_time
    token_count = _count_tokens(final_messages)

    print(
        f"[Stats] Total rounds: {agent.state.step}, "
        f"Tool call rounds: {agent.state.tool_rounds}, "
        f"Time: {total_process_time:.1f}s"
    )

    return {
        "question": question,
        "answer": answer,
        "messages": final_messages,
        "prediction": prediction,
        "termination": termination,
        "token_count": token_count,
        "total_process_time": total_process_time,
        "total_rounds": agent.state.step,
        "tool_call_rounds": agent.state.tool_rounds,
    }
