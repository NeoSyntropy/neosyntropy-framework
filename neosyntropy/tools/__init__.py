"""Tool registry with pydantic args and invocation audit logging.



Built-in adapters (Agno-style) live alongside the registry, e.g.

:mod:`neosyntropy.tools.emailjs`, :mod:`neosyntropy.tools.ast_tools`,

and :mod:`neosyntropy.tools.coding_tools`.

"""



from .ast_tools import (

    AnalyzeFileArgs,

    AstAnalyzer,

    AstTools,

    AstToolsError,

    FindBareExceptsArgs,

)

from .coding_tools import (

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

from .emailjs import (

    EmailJSClient,

    EmailJSCredentials,

    EmailJSError,

    EmailJSTools,

    SendEmailArgs,

)

from .registry import (

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

    "EmailJSClient",

    "EmailJSCredentials",

    "EmailJSError",

    "EmailJSTools",

    "FindArgs",

    "FindBareExceptsArgs",

    "GrepArgs",

    "LsArgs",

    "ReadFileArgs",

    "RegisteredTool",

    "RunShellArgs",

    "SendEmailArgs",

    "ToolInvocation",

    "ToolNotAllowedError",

    "ToolRegistry",

    "WriteFileArgs",

    "neosyntropy",

    "registered_tools",

    "tool",

]


