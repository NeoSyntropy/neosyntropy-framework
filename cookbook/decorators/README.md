# Decorators cookbook

Turn a Python function into a controlled LLM call. The model extracts
typed parameters; your function runs with those values.

| Decorator | When to use |
|---|---|
| `@function_calling` | One extraction step. No tools. |
| `@workflow` | Reasoning steps call tools first, then extract parameters. |

Credentials: copy [`tests/.env.example`](../../tests/.env.example) to
`tests/.env` (or set the same variables in the environment).

```bash
python cookbook/decorators/function_calling_example.py
python cookbook/decorators/workflow_reasoning_example.py
```

## `@function_calling`

A single `SchemaNode` maps natural language onto the function's pydantic
parameter model, then calls the function.

```python
@function_calling(
    prompt="Extract the greeting name and language from the request.",
    input_schema=UserRequest,
    client=client,
)
def greet(params: GreetParams) -> str:
    return f"Hello, {params.name}!"
```

## `@workflow`

Each `ReasoningStep` is an allow-listed tool step. After the tools return
evidence, a `SchemaNode` predicts the function parameters.

```python
@workflow(
    input_schema=UserRequest,
    steps=[
        ReasoningStep("Identify the product SKU.", tools=["lookup_sku"]),
        ReasoningStep("Check warehouse stock.", tools=["check_stock"]),
        SchemaStep(),
    ],
    client=client,
    tools=registry,
)
def place_order(params: OrderParams) -> str:
    return f"Ordered {params.quantity} x {params.sku}"
```
