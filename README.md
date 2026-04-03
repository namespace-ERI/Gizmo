# Gizmo

Gizmo is a lightweight agent framework for building tool-using research pipelines on top of OpenAI-compatible chat APIs and the OpenAI Responses API. It provides:

- a reusable agent runtime with step limits, timeout limits, and trajectory recording
- a `GPTAgent` adapter that talks to the official OpenAI Responses API with native function tools
- a Qwen-oriented agent adapter that uses XML-style tool calls plus optional `<think>` reasoning blocks
- a `GPTOssAgent` adapter aligned to the `gpt-oss` chat template served from vLLM
- lifecycle hooks for injecting custom logic around each LLM/tool step
- context-manager middleware for message rebuilding, truncation, token budgeting, and other pre-LLM transforms
- built-in online and offline search/visit tools

The current design keeps the runtime generic and leaves task-specific strategy to the pipeline layer. If you need memory injection, shared state updates, or custom stop policies, you implement them with hooks and context managers instead of rewriting the full loop.

## What Is Implemented

### Runtime

- `BaseAgent`: generic tool-using agent loop
- `LLMConfig`: per-agent LLM generation config
- `RunState`: per-run mutable runtime state
- `ContextManager`: pluggable message transformation middleware
- lifecycle hooks: `before_llm`, `after_llm`, `after_tool`, `should_stop`
- built-in stopping controls: `max_steps`, `max_time_seconds`, `max_tool_rounds`
- trajectory inspection via `print_trajectory()` and `save_trajectory()`

### Agent Adapters

- `GPTAgent`: OpenAI Responses API adapter with native function calling
- official function-tool schema and `function_call_output` reinjection
- `QwenAgent`: prompt-based Qwen adapter
- XML tool-call prompt construction
- `<think>...</think>` reasoning extraction
- prompt-based multi-tool parsing
- tool-result reinjection as `<tool_response>...</tool_response>`
- `GPTOssAgent`: native tool-calling adapter aligned to the `gpt-oss` chat template

### Built-in Tools

- `EchoTool`: minimal smoke-test tool
- `SearchTool`: batched web search through Serper
- `VisitTool`: Jina Reader fetch + auxiliary LLM evidence extraction
- `LocalSearchTool`: FAISS-backed local corpus search with remote embedding API
- `LocalVisitTool`: local corpus page retrieval by `docid` + auxiliary LLM extraction

## Package Layout

```text
Gizmo/
  Gizmo/
    agents/
      base_agent.py
      gpt_agent.py
      gpt_oss_agent.py
      qwen_agent.py
    prompts/
      system_prompt.py
      tool_prompt.py
    tools/
      base_tool.py
      echo_tool.py
      search_tool.py
      visit_tool.py
      local_search_tool.py
      local_visit_tool.py
    utils/
      message_parser.py
  README.md
  pyproject.toml
```

## Installation

From the `Gizmo/` directory:

```bash
pip install -e .
```

`pyproject.toml` currently only defines the package itself and does not declare runtime dependencies, so install the libraries you need manually.

Minimum runtime for the core agent:

```bash
pip install openai
```

Additional dependencies by feature:

```bash
pip install requests tiktoken
pip install numpy datasets faiss-cpu
```

Notes:

- Use `faiss-gpu` instead of `faiss-cpu` if that matches your environment.
- `SearchTool` requires a Serper API key.
- `VisitTool` requires a Jina Reader token plus an auxiliary chat model endpoint.
- `LocalSearchTool` requires a local FAISS index, local Arrow corpus files, and an embedding endpoint.
- `LocalVisitTool` requires a chat model endpoint plus a linked `LocalSearchTool`.

## Core Concepts

### `BaseAgent`

`BaseAgent` owns the generic execution loop:

1. reset messages, trajectory, and `RunState`
2. append the user message
3. check stop conditions
4. fire `before_llm`
5. apply registered `ContextManager`s inside the LLM call path
6. call the chat model
7. parse reasoning, tool calls, and final content
8. fire `after_llm`
9. execute tools if requested
10. fire `after_tool` for each tool call
11. append tool-response messages and continue

If a built-in stop condition fires, Gizmo can inject one more user prompt asking the model for a best-effort final answer.

### `RunState`

`RunState` is reset on every `run()` / `run_verbose()` call and exposes:

- `step`: completed LLM rounds
- `tool_rounds`: rounds that executed at least one tool
- `elapsed`: seconds since the run started
- `stop_reason`: `""`, `max_steps`, `timeout`, `max_tool_rounds`, or your custom stop reason

Use it for:

- custom stop logic
- tracing and logging
- coordinating context managers
- carrying lightweight per-run control information

### `LLMConfig`

`LLMConfig` controls both OpenAI-compatible chat generation parameters and the Responses API fields consumed by `GPTAgent`:

- `max_tokens`
- `max_output_tokens`
- `temperature`
- `top_p`
- `seed`
- `timeout`
- `enable_thinking`
- `extra_body`
- `store`
- `truncation`
- `parallel_tool_calls`
- `tool_choice`
- `reasoning` / `reasoning_effort` / `reasoning_summary`
- `text` / `text_verbosity` / `text_format`
- `include`
- `metadata`
- `service_tier`
- `prompt_cache_key`
- `safety_identifier`

`enable_thinking=True` injects `chat_template_kwargs.enable_thinking=True` into `extra_body`.
For `GPTAgent`, prefer the native Responses API knobs such as `max_output_tokens`, `parallel_tool_calls`, `reasoning_effort`, `reasoning_summary`, and `text_verbosity`.

### `ContextManager`

`ContextManager` is a middleware interface:

```python
from Gizmo.agents.base_agent import ContextManager, RunState


class ExampleContextManager(ContextManager):
    def process(self, messages: list[dict], state: RunState) -> list[dict]:
        return messages

    def reset(self) -> None:
        pass
```

Use it for:

- token budgeting
- history trimming
- prompt rebuilding
- truncation notices
- memory injection
- shared-state context augmentation

`process()` receives the full message list including the system message and must return the transformed list.

### Lifecycle Hooks

Hooks let you attach runtime behavior without subclassing the full loop:

- `before_llm(state, messages)`: mutate or inspect the current conversation history before the API call
- `after_llm(state, parsed)`: inspect parsed model output
- `after_tool(state, tool_name, tool_args, tool_result)`: react after each tool execution
- `should_stop(state) -> Optional[str]`: return a non-empty stop reason to halt the run

Typical use cases:

- `before_llm`: inject memory, rebuild the first user turn, add control messages
- `after_llm`: detect malformed output, log parsed tool calls, collect traces
- `after_tool`: update memory pages, caches, or shared state
- `should_stop`: enforce domain-specific limits or quality gates

## Quickstart

Minimal example using the built-in `EchoTool`:

```python
from Gizmo.agents.base_agent import LLMConfig
from Gizmo.agents.qwen_agent import QwenAgent
from Gizmo.tools.echo_tool import EchoTool


agent = QwenAgent(
    model="Qwen3.5-122B-A10B",
    api_key="EMPTY",
    base_url="http://localhost:8001/v1",
    tools=[EchoTool()],
    max_steps=20,
    llm_config=LLMConfig(
        max_tokens=8192,
        temperature=0.7,
        seed=42,
        timeout=120.0,
        enable_thinking=True,
    ),
)

result = agent.run("Please call echo and return hello")
print(result)
agent.print_trajectory()
agent.save_trajectory("logs/trajectory.json")
```

Official OpenAI Responses API example:

```python
from Gizmo.agents.base_agent import LLMConfig
from Gizmo.agents.gpt_agent import GPTAgent
from Gizmo.tools.echo_tool import EchoTool


agent = GPTAgent(
    model="gpt-5-mini",
    api_key="YOUR_OPENAI_API_KEY",
    tools=[EchoTool()],
    max_steps=20,
    llm_config=LLMConfig(
        max_output_tokens=4096,
        temperature=0.7,
        parallel_tool_calls=True,
        reasoning_effort="medium",
        reasoning_summary="auto",
        text_verbosity="medium",
    ),
)

result = agent.run("Please call echo and return hello")
print(result)
```

## Hook and Context Example

This is the intended extension pattern for pipeline-level features such as memory and context control:

```python
from Gizmo.agents.base_agent import ContextManager, LLMConfig, RunState
from Gizmo.agents.qwen_agent import QwenAgent
from Gizmo.tools.echo_tool import EchoTool


class ExampleContextManager(ContextManager):
    def process(self, messages: list[dict], state: RunState) -> list[dict]:
        # Insert token budgeting, truncation, memory injection, or message rebuild here.
        return messages

    def reset(self) -> None:
        pass


agent = QwenAgent(
    model="Qwen3.5-122B-A10B",
    api_key="EMPTY",
    base_url="http://localhost:8001/v1",
    tools=[EchoTool()],
    max_steps=20,
    max_time_seconds=300.0,
    max_tool_rounds=15,
    llm_config=LLMConfig(
        max_tokens=8192,
        temperature=0.7,
        seed=42,
        timeout=120.0,
        enable_thinking=True,
    ),
)

agent.use(ExampleContextManager())

agent.on("before_llm", lambda s, msgs: print(f"[Hook] before_llm step={s.step}"))
agent.on("after_llm", lambda s, parsed: print(f"[Hook] after_llm final={bool(parsed.get('final_content'))}"))
agent.on("after_tool", lambda s, name, args, result: print(f"[Hook] after_tool {name}"))
agent.on("should_stop", lambda s: "custom_limit" if s.tool_rounds >= 10 else None)

print(agent.run("Please call echo and return hello"))
```

If you return a custom stop reason from `should_stop`, `BaseAgent` currently returns `[stopped: your_reason]` unless you override `_on_stop()` in a subclass to inject a final-answer prompt.

## QwenAgent Behavior

`QwenAgent` is the main built-in adapter. It:

- builds a system prompt that embeds the available tool definitions
- instructs the model to output tool calls in XML
- extracts reasoning from `<think>...</think>`
- extracts tool calls from `<tool_call>...</tool_call>`
- strips reasoning and tool XML when computing `final_content`
- appends raw assistant output back into the conversation so the model can see its previous tool-call text

Important parser note:

- the current built-in `QwenAgent` parser keeps parameter bodies as strings
- if you need automatic JSON deserialization for arrays or objects, extend `QwenAgent` or wire in `Gizmo.utils.message_parser`
- scalar string parameters work out of the box

Expected tool-call format:

```xml
<tool_call>
<function=search>
<parameter=query>
example query
</parameter>
</function>
</tool_call>
```

Tool results are fed back as:

```xml
<tool_response>
...tool output...
</tool_response>
Please continue.
```

## GPTAgent Behavior

`GPTAgent` is the OpenAI-first adapter. It:

- calls `client.responses.create(...)` instead of `chat.completions.create(...)`
- sends the system prompt as Responses API `instructions`
- converts runtime history into official Responses API `input` items
- uses official function tools with top-level `type` / `name` / `description` / `parameters`
- appends returned `response.output` items back into history for the next turn
- sends local Python tool results back as `function_call_output` items
- keeps the existing hook names, stop controls, trajectory recording, and `run()` / `run_verbose()` interface

Compatibility notes:

- `GPTAgent` history is stored as Responses API input items, not only as chat-style `role/content` messages.
- `before_llm` hooks and `ContextManager`s still work, but they may now see a mixed list containing `message`, `function_call`, `reasoning`, and `function_call_output` items.
- If a hook appends plain chat-style items such as `{"role": "user", "content": "..."}`, `GPTAgent` will normalize them into Responses API `message` items automatically.

## Built-in Tools

When you call tool classes directly in Python, several of them accept structured inputs such as Python lists. In the default prompt-based `QwenAgent` path, complex parameter bodies are still delivered as raw strings unless you customize the parser.

### `EchoTool`

Smallest possible tool for smoke tests and parser verification.

```python
from Gizmo.tools.echo_tool import EchoTool

tool = EchoTool()
print(tool.execute(input="hello"))
```

### `SearchTool`

Online batched search over Serper.

Constructor:

```python
from Gizmo.tools.search_tool import SearchTool

tool = SearchTool(api_key="SERPER_API_KEY")
```

Behavior:

- accepts `query` as a string or list of strings
- runs queries in parallel with `ThreadPoolExecutor`
- auto-switches search locale between Chinese and US English
- returns formatted web results with title, link, optional date/source, and snippet

### `VisitTool`

Online webpage reader built on Jina Reader plus an auxiliary LLM.

Constructor:

```python
from Gizmo.tools.visit_tool import VisitTool

tool = VisitTool(
    jina_api_key="JINA_API_KEY",
    llm_api_key="EMPTY",
    llm_base_url="http://localhost:8001/v1",
    llm_model="Qwen3.5-32B",
)
```

Behavior:

- accepts `url` as a string or list
- requires a `goal`
- fetches page markdown via `https://r.jina.ai/`
- truncates oversized content by token count
- uses an auxiliary LLM to extract JSON with evidence and summary
- retries short or malformed extraction outputs

### `LocalSearchTool`

Offline/local corpus search using a FAISS index and a remote embedding endpoint.

Constructor:

```python
from Gizmo.tools.local_search_tool import LocalSearchTool

search_tool = LocalSearchTool(
    index_path="path/to/index_dir",
    corpus_path="path/to/arrow_dir",
    embed_api_url="http://localhost:8001/v1",
    embed_model="Qwen3-Embedding-0.6B",
    embed_api_key="EMPTY",
    top_k=5,
)
```

Behavior:

- lazy-loads `corpus.index` and `corpus_lookup.pkl`
- loads Arrow corpora from disk
- requests query embeddings from an OpenAI-compatible embedding endpoint
- returns session-scoped integer `docid`s
- caches `docid -> url` and `docid -> full text` for later retrieval

### `LocalVisitTool`

Offline/local document reader that depends on `LocalSearchTool`.

Constructor:

```python
from Gizmo.tools.local_search_tool import LocalSearchTool
from Gizmo.tools.local_visit_tool import LocalVisitTool

search_tool = LocalSearchTool(
    index_path="path/to/index_dir",
    corpus_path="path/to/arrow_dir",
    embed_api_url="http://localhost:8001/v1",
    embed_model="Qwen3-Embedding-0.6B",
)

visit_tool = LocalVisitTool(
    llm_api_key="EMPTY",
    llm_base_url="http://localhost:8001/v1",
    llm_model="Qwen3.5-32B",
    search_tool=search_tool,
)
```

Behavior:

- accepts one or more `docid`s returned by `LocalSearchTool`
- looks up cached local document text instead of making network fetches
- requires a `goal`
- uses an auxiliary LLM to extract targeted evidence
- returns an error if the `docid` does not exist in the current session cache

## End-to-End Research Example

Online research stack:

```python
from Gizmo.agents.base_agent import LLMConfig
from Gizmo.agents.qwen_agent import QwenAgent
from Gizmo.tools.search_tool import SearchTool
from Gizmo.tools.visit_tool import VisitTool


agent = QwenAgent(
    model="Qwen3.5-122B-A10B",
    api_key="EMPTY",
    base_url="http://localhost:8001/v1",
    tools=[
        SearchTool(api_key="SERPER_API_KEY"),
        VisitTool(
            jina_api_key="JINA_API_KEY",
            llm_api_key="EMPTY",
            llm_base_url="http://localhost:8001/v1",
            llm_model="Qwen3.5-32B",
        ),
    ],
    max_steps=20,
    max_time_seconds=300.0,
    max_tool_rounds=15,
    llm_config=LLMConfig(max_tokens=8192, enable_thinking=True),
)

answer = agent.run("Research the topic and cite evidence from visited pages.")
print(answer)
```

Offline/local corpus stack:

```python
from Gizmo.agents.base_agent import LLMConfig
from Gizmo.agents.qwen_agent import QwenAgent
from Gizmo.tools.local_search_tool import LocalSearchTool
from Gizmo.tools.local_visit_tool import LocalVisitTool


search_tool = LocalSearchTool(
    index_path="path/to/index_dir",
    corpus_path="path/to/arrow_dir",
    embed_api_url="http://localhost:8001/v1",
    embed_model="Qwen3-Embedding-0.6B",
)

visit_tool = LocalVisitTool(
    llm_api_key="EMPTY",
    llm_base_url="http://localhost:8001/v1",
    llm_model="Qwen3.5-32B",
    search_tool=search_tool,
)

agent = QwenAgent(
    model="Qwen3.5-122B-A10B",
    api_key="EMPTY",
    base_url="http://localhost:8001/v1",
    tools=[search_tool, visit_tool],
    max_steps=20,
    llm_config=LLMConfig(max_tokens=8192, enable_thinking=True),
)

answer = agent.run("Search the local corpus and then open relevant pages by docid.")
print(answer)
```

## Writing Custom Tools

Subclass `BaseTool` and implement `execute()`:

```python
from Gizmo.tools.base_tool import BaseTool


class MyTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="Do a custom action.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Input query"},
                },
                "required": ["query"],
            },
        )

    def execute(self, **kwargs) -> str:
        return f"handled: {kwargs['query']}"
```

Then pass it into the agent:

```python
agent = QwenAgent(
    model="Qwen3.5-122B-A10B",
    api_key="EMPTY",
    base_url="http://localhost:8001/v1",
    tools=[MyTool()],
)
```

## Return Values and Introspection

### `run()`

Returns the final assistant content as a string.

### `run_verbose()`

Returns the parsed result dictionary, which may contain:

- `assistant_message`
- `output_items`
- `tool_calls`
- `reasoning_content`
- `final_content`
- `response_id`
- `response_status`

### `trajectory`

Each completed step is recorded as a `TrajectoryStep` with:

- `step`
- `reasoning`
- `tool_calls`
- `final_content`

### `save_trajectory(path)`

Writes the accumulated runtime history as JSON.
For chat-style agents this is a message list with the system prompt; for `GPTAgent` it is a Responses-style payload with `instructions` and `input`.

## Current Limitations

- `pyproject.toml` does not yet declare runtime dependencies.
- Tool re-exports are still minimal; importing from concrete module paths such as `Gizmo.agents.base_agent` or `Gizmo.tools.search_tool` remains the clearest option.
- `QwenAgent` uses prompt-based XML tool calling, not OpenAI native tool calling.
- `QwenAgent` does not currently deserialize XML parameter bodies into Python lists or dicts; structured tool args need a custom parser integration.
- `GPTAgent` currently executes locally registered Python `function` tools; if you want hosted OpenAI tools such as web search or file search, wire them into your project flow explicitly.
- No built-in memory manager is included in the package; memory, shared state, and context policies are expected to be implemented with hooks and context managers.
- `mock_llm.py` is currently just a placeholder.

## Recommended Extension Boundary

Keep these concerns inside Gizmo:

- generic run loop
- lifecycle hooks
- context manager pipeline
- tool abstraction
- trajectory recording
- generic stopping and finalization

Keep these concerns in your pipeline or project layer:

- memory-page schemas
- shared-state formats
- task-specific prompt rebuilding
- domain-specific stop rules
- evaluation output formats
- task-specific tools and post-processing

That split keeps Gizmo reusable while still letting your pipeline implement richer behaviors such as memory injection, truncation notices, retrieval caches, and shared multi-agent coordination.
