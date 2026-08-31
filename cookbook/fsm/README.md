# FSM cookbook

Standalone examples for the core FSM authoring primitives.

Each script loads `tests/.env`, creates its own project, and runs against the
local backend by default.

## Examples

- `schema_node_example.py` - simple schema extraction with `SchemaNode`
- `reasoning_node_prompt_tools_example.py` - reasoning node with prompt + tools
- `reasoning_node_steps_example.py` - reasoning flow with step-by-step instructions
- `semantic_router_parallel_example.py` - semantic router with independent branches
- `semantic_router_sequential_example.py` - semantic router with a chained follow-up step

## Run

```bash
python cookbook/fsm/schema_node_example.py
python cookbook/fsm/reasoning_node_prompt_tools_example.py
python cookbook/fsm/reasoning_node_steps_example.py
python cookbook/fsm/semantic_router_parallel_example.py
python cookbook/fsm/semantic_router_sequential_example.py
```

If the local GPU inference service is unavailable, set `NEOSYNTROPY_PROVIDER`
to a hosted model such as `gemini-2.5-flash`.
