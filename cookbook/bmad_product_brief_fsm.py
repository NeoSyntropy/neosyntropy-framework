import os
from pydantic import BaseModel, ConfigDict, Field
from neosyntropy import (
    Client,
    FSM,
    OpenInput,
    ReasoningNode,
    SchemaNode,
    TextOutput,
    ToolRegistry,
    tool,
    edge_deterministic,
)
from neosyntropy.routing.declarations import SemanticRouter, DeterministicRouter, compile_routers

VERTEX_MODEL = "gemini-2.5-flash"
registry = ToolRegistry()

# ---------------------------------------------------------
# 1. Input Schemas
# ---------------------------------------------------------

class SkillActivation(BaseModel):
    """Input payload for activating the bmad-product-brief skill."""
    model_config = ConfigDict(extra="forbid")
    user_request: str
    is_headless: bool = False
    project_workspace: str

# ---------------------------------------------------------
# 2. Tools (Replacing BMAD Bash Scripts)
# ---------------------------------------------------------

class MemlogArgs(BaseModel):
    workspace: str
    entry_type: str = Field(description="One of: decision, change, override, assumption, event")
    text: str

@tool(registry=registry)
def append_memlog(args: MemlogArgs) -> dict:
    """Append a decision or event to the .memlog.md audit trail."""
    print(f"[TOOL] Appending {args.entry_type} to memlog: {args.text}")
    return {"status": "success", "appended": True}

class WebSearchArgs(BaseModel):
    query: str

@tool(registry=registry)
def web_search(args: WebSearchArgs) -> dict:
    """Perform web research to gather landscape and comparable data."""
    print(f"[TOOL] Searching web for: {args.query}")
    return {"status": "success", "results": "Simulated search results: Product landscape looks promising."}

class ReadFileArgs(BaseModel):
    filepath: str

@tool(registry=registry)
def read_workspace_file(args: ReadFileArgs) -> dict:
    """Read contents of a file (e.g. existing brief.md or addendum.md) in the workspace."""
    print(f"[TOOL] Reading file: {args.filepath}")
    return {"status": "success", "content": "# Simulated Brief Content"}

# ---------------------------------------------------------
# 3. Nodes (The actual execution states)
# ---------------------------------------------------------

# Node for Discovery and Coaching (Interactive Mode)
discovery_node = ReasoningNode(
    id="DiscoveryPhase",
    input_schema=OpenInput,
    provider=VERTEX_MODEL,
    prompt=(
        "You are an expert product analyst coach. The user wants to craft a product brief. "
        "Conversationally surface what the user brings, why this brief exists, the domain, and the form-factor. "
        "Use tools to append to the memlog, do web research on comparables, and check existing files. "
        "Ask clarifying questions and do not do the thinking for them. Let them sweat if assumptions are unexamined."
    ),
    tools=("append_memlog", "web_search", "read_workspace_file"),
)

# Node for Headless Override
headless_processing_node = ReasoningNode(
    id="HeadlessProcessing",
    input_schema=OpenInput,
    provider=VERTEX_MODEL,
    prompt=(
        "You are in HEADLESS mode. Do not ask questions. "
        "Complete the product brief update, creation, or validation using only the provided context. "
        "Use tools to check existing files and write to memlog."
    ),
    tools=("append_memlog", "web_search", "read_workspace_file"),
)

# Node for Finalizing output
class FinalizeOutput(BaseModel):
    status: str
    intent: str
    brief_path: str
    addendum_path: str | None = None
    open_questions: list[str] = []

finalize_node = SchemaNode(
    id="FinalizeNode",
    input_schema=OpenInput,
    output_schema=FinalizeOutput,
    provider=VERTEX_MODEL,
    prompt=(
        "Generate the final summary metadata of the completed brief operation. "
        "This acts as the JSON return payload for the headless mode or the final status block."
    ),
)

# ---------------------------------------------------------
# 4. Routers (Replacing LLM Prompt-based Intent Checks)
# ---------------------------------------------------------

# Semantic router to determine intent
intent_router = SemanticRouter(
    id="IntentRouter",
    input_schema=SkillActivation,
    routes={
        "create_new_brief": discovery_node,
        "update_existing_brief": discovery_node, 
        "validate_brief": discovery_node,        
    },
    fallback_node=discovery_node, 
)

# Deterministic router to branch on 'headless' flag
mode_router = DeterministicRouter(
    id="ModeRouter",
    input_schema=SkillActivation,
    rules=[
        (lambda ctx: ctx.state.get("is_headless") is True, headless_processing_node),
        (lambda ctx: ctx.state.get("is_headless") is False, intent_router),
    ]
)

# ---------------------------------------------------------
# 5. Graph Definition
# ---------------------------------------------------------

# Compile routers into edges
router_edges = compile_routers([mode_router, intent_router])

# Link Discovery and Headless nodes to Finalize Node
flow_edges = [
    edge_deterministic("DiscoveryPhase", "FinalizeNode"),
    edge_deterministic("HeadlessProcessing", "FinalizeNode"),
    edge_deterministic("FinalizeNode", "End")
]

# Wire it all up into a strict NeoSyntropy FSM
fsm = FSM(
    entry=mode_router,
    nodes=[discovery_node, headless_processing_node, finalize_node],
    edges=router_edges + flow_edges,
)

if __name__ == "__main__":
    print("Proof of concept for migrating BMAD Skills FSM to NeoSyntropy Framework is defined.")
    # test run would look like: fsm.run(SkillActivation(...), state={...}, client=...)
