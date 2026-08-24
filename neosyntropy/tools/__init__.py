"""Tool registry with pydantic args and invocation audit logging.



Built-in adapters (NeoSyntropy-style) live alongside the registry, e.g.

:mod:`neosyntropy.tools.email.email`, :mod:`neosyntropy.tools.coding.ast_tools`,

and :mod:`neosyntropy.tools.coding.coding_tools`.

"""



from .coding.ast_tools import (

    AnalyzeFileArgs,

    AstAnalyzer,

    AstTools,

    AstToolsError,

    FindBareExceptsArgs,

)

from .core.toolkit import Toolkit

from .coding.coding_tools import (

    CodingTools,

    CodingToolsError,

    CodingWorkspace,

    DEFAULT_ALLOWED_COMMANDS,

    EditFileArgs,

    FindArgs,

    GrepArgs,

    LsArgs,

    ReadFileArgs,

    RunShellArgs,

    WriteFileArgs,

)



from .core.registry import (

    DEFAULT_REGISTRY,

    BoundTools,

    RegisteredTool,

    ToolInvocation,

    ToolNotAllowedError,

    ToolRegistry,

    neosyntropy,

    registered_tools,

    tool,

)



__all__ = [

    "Toolkit",

    "AnalyzeFileArgs",

    "AstAnalyzer",

    "AstTools",

    "AstToolsError",

    "CodingTools",

    "CodingToolsError",

    "CodingWorkspace",

    "DEFAULT_ALLOWED_COMMANDS",

    "DEFAULT_REGISTRY",

    "BoundTools",

    "EditFileArgs",

    "FindArgs",

    "FindBareExceptsArgs",

    "GrepArgs",

    "LsArgs",

    "ReadFileArgs",

    "RegisteredTool",

    "RunShellArgs",

    "ToolInvocation",

    "ToolNotAllowedError",

    "ToolRegistry",

    "WriteFileArgs",

    "neosyntropy",

    "registered_tools",

    "tool",

]


