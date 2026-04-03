"""
run_task_gpt_bc.py - Evaluation runner for the Gizmo-based GPTAgent
on the BrowseComp (BC) dataset.

Uses demo/single_agent_gpt.run_react_agent which is built on the Gizmo
framework with GPTAgent + SearchTool + VisitTool.
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace
from typing import Optional


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_WORKSPACE_ROOT = os.path.dirname(_REPO_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from single_agent_gpt import run_react_agent
from Gizmo.tools.search_tool import SearchTool


_DEFAULT_EVAL_DATA_PATH = os.path.join(
    _WORKSPACE_ROOT, "GEM", "data", "BrowseComp", "eval_bc_100.jsonl"
)
_DEFAULT_RESULTS_DIR = os.path.join(
    _WORKSPACE_ROOT, "GEM", "data", "BrowseComp", "results"
)
_DEFAULT_LOGS_BASE_DIR = os.path.join(
    _WORKSPACE_ROOT, "GEM", "data", "BrowseComp", "logs"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model", type=str, default="gpt-5-mini")
    parser.add_argument("--reasoning_model_api_key", type=str, default="")
    parser.add_argument(
        "--reasoning_model_base_url",
        type=str,
        default="https://api.openai.com/v1",
    )
    parser.add_argument("--auxiliary_model", type=str, default="")
    parser.add_argument("--auxiliary_model_api_key", type=str, default="")
    parser.add_argument(
        "--auxiliary_model_base_url",
        type=str,
        default="https://api.openai.com/v1",
    )
    parser.add_argument(
        "--search_api_key",
        "--serper_api_key",
        dest="search_api_key",
        type=str,
        default="",
    )
    parser.add_argument("--jina_api_key", type=str, default="")
    parser.add_argument("--eval_task", type=str, default="bc")
    parser.add_argument("--eval_data_path", type=str, default=_DEFAULT_EVAL_DATA_PATH)
    parser.add_argument("--results_dir", type=str, default=_DEFAULT_RESULTS_DIR)
    parser.add_argument("--logs_base_dir", type=str, default=_DEFAULT_LOGS_BASE_DIR)
    parser.add_argument("--version", type=str, default="gpt_demo_bc")
    parser.add_argument("--max_llm_call_per_run", type=int, default=150)
    parser.add_argument("--question_workers", type=int, default=4)
    parser.add_argument("--max_time_seconds", type=float, default=150 * 60)
    parser.add_argument("--llm_timeout", type=float, default=120.0)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=105 * 1024,
        help="Approximate live-context token budget before forced finalization.",
    )
    parser.add_argument("--max_output_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--parallel_tool_calls",
        action="store_true",
        help="Allow the model to emit multiple function calls in one Responses turn.",
    )
    parser.add_argument("--reasoning_effort", type=str, default="medium")
    parser.add_argument("--reasoning_summary", type=str, default="auto")
    parser.add_argument("--text_verbosity", type=str, default="medium")
    return parser


def build_config(args: argparse.Namespace) -> SimpleNamespace:
    llm_generate_cfg = {
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    llm_generate_cfg = {
        key: value for key, value in llm_generate_cfg.items() if value is not None
    }
    return SimpleNamespace(**vars(args), llm_generate_cfg=llm_generate_cfg)


def load_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def process_item_bc(
    item: dict,
    config,
    corpus_searcher: SearchTool,
    log_dir: Optional[str] = None,
) -> dict:
    question = item["question"]
    answer = item.get("golden_answers", "")
    item_id = item.get("id", "")
    try:
        result = run_react_agent(
            question,
            answer,
            config,
            corpus_searcher=corpus_searcher,
            log_dir=log_dir,
            log_name=item_id,
        )
        return {"id": item_id, **result}
    except Exception as e:
        print(f"Error processing item {item_id}: {e}")
        return {
            "id": item_id,
            "question": question,
            "answer": answer,
            "error": str(e),
        }


def run_evaluation(config) -> None:
    if str(config.eval_task).lower() != "bc":
        raise ValueError("run_task_gpt_bc.py currently supports only eval_task='bc'.")
    if not config.search_api_key:
        raise ValueError(
            "run_task_gpt_bc.py requires --search_api_key (or --serper_api_key)."
        )
    if not config.jina_api_key:
        raise ValueError("run_task_gpt_bc.py requires --jina_api_key.")
    if not config.reasoning_model_api_key:
        raise ValueError("run_task_gpt_bc.py requires --reasoning_model_api_key.")

    model_tag = config.reasoning_model.split("/")[-1]

    print("Initializing SearchTool ...")
    corpus_searcher = SearchTool(api_key=config.search_api_key)

    eval_data_path = config.eval_data_path or _DEFAULT_EVAL_DATA_PATH
    results_dir = config.results_dir or _DEFAULT_RESULTS_DIR
    logs_base_dir = config.logs_base_dir or _DEFAULT_LOGS_BASE_DIR

    run_tag = f"{model_tag}_{config.version}_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(logs_base_dir, run_tag)
    os.makedirs(log_dir, exist_ok=True)

    os.makedirs(results_dir, exist_ok=True)
    eval_data = load_jsonl(eval_data_path)
    result_path = os.path.join(
        results_dir, f"eval_results_{model_tag}_{config.version}.jsonl"
    )

    print("Running BC evaluation (Gizmo / GPTAgent)")
    print(f"Eval Data -> {eval_data_path}")
    print(f"Processing {len(eval_data)} items with {config.question_workers} workers")
    print(f"Results  -> {result_path}")
    print(f"Logs     -> {log_dir}/")

    process_func = partial(
        process_item_bc,
        config=config,
        corpus_searcher=corpus_searcher,
        log_dir=log_dir,
    )

    completed_count = 0
    with open(result_path, "w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=config.question_workers) as executor:
            futures = [executor.submit(process_func, item) for item in eval_data]
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                try:
                    result = future.result()
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    completed_count += 1
                    print(f"Completed {i}/{len(eval_data)} items")
                except Exception as e:
                    print(f"Error retrieving future result: {e}")

    print(
        f"Evaluation complete. {completed_count}/{len(eval_data)} items written to "
        f"{result_path}"
    )


if __name__ == "__main__":
    parser = build_parser()
    config = build_config(parser.parse_args())

    print("=" * 60)
    print("ReAct Agent Evaluation Configuration (Gizmo / GPTAgent / BC)")
    print("=" * 60)
    print("  Task:             bc")
    print(f"  Version:          {config.version}")
    print(f"  Reasoning Model:  {config.reasoning_model}")
    print(f"    URL:            {config.reasoning_model_base_url}")
    print(f"  Auxiliary Model:  {config.auxiliary_model or '(same as reasoning)'}")
    print(
        f"    URL:            "
        f"{config.auxiliary_model_base_url or '(same as reasoning)'}"
    )
    print(
        "  Search API Key:   "
        f"{'configured' if config.search_api_key else '(not set)'}"
    )
    print(
        "  Jina API Key:     "
        f"{'configured' if config.jina_api_key else '(not set)'}"
    )
    print("  Eval Data:        " f"{config.eval_data_path or _DEFAULT_EVAL_DATA_PATH}")
    print("  Results Dir:      " f"{config.results_dir or _DEFAULT_RESULTS_DIR}")
    print("  Logs Base Dir:    " f"{config.logs_base_dir or _DEFAULT_LOGS_BASE_DIR}")
    print(f"  Max LLM Calls:    {config.max_llm_call_per_run}")
    print(f"  Question Workers: {config.question_workers}")
    print(f"  Max Output Tok:   {config.max_output_tokens}")
    print(f"  Reasoning Effort: {config.reasoning_effort or '(default)'}")
    print(f"  Text Verbosity:   {config.text_verbosity or '(default)'}")
    print(f"  Parallel Tools:   {config.parallel_tool_calls}")
    print("=" * 60)
    print()

    run_evaluation(config)
