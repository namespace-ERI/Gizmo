#!/usr/bin/env bash
set -euo pipefail

cd /share/project/yuyang/Gizmo/demo

# Edit these values directly before running.
REASONING_MODEL="gpt-5.4-mini"
REASONING_MODEL_BASE_URL=""
REASONING_MODEL_API_KEY=""

AUXILIARY_MODEL="gpt-5.4-mini"
AUXILIARY_MODEL_BASE_URL=""
AUXILIARY_MODEL_API_KEY=""

SEARCH_API_KEY=""
JINA_API_KEY=""

python3 run_task_gpt_bc.py \
  --eval_task bc \
  --eval_data_path /share/project/yuyang/Gizmo/demo/eval_bc_100.jsonl \
  --results_dir /share/project/yuyang/Gizmo/demo/results \
  --logs_base_dir /share/project/yuyang/Gizmo/demo/logs \
  --reasoning_model "${REASONING_MODEL}" \
  --reasoning_model_base_url "${REASONING_MODEL_BASE_URL}" \
  --reasoning_model_api_key "${REASONING_MODEL_API_KEY}" \
  --auxiliary_model "${AUXILIARY_MODEL}" \
  --auxiliary_model_base_url "${AUXILIARY_MODEL_BASE_URL}" \
  --auxiliary_model_api_key "${AUXILIARY_MODEL_API_KEY}" \
  --search_api_key "${SEARCH_API_KEY}" \
  --jina_api_key "${JINA_API_KEY}" \
  --question_workers 4 \
  --max_output_tokens 8192 \
  --parallel_tool_calls \
  --reasoning_effort medium \
  --reasoning_summary auto \
  --text_verbosity medium \
  --version single_512_top5_gpt_gizmo_bc_eval_bc_100
