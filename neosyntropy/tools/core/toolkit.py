import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from pydantic import create_model, BaseModel
from neosyntropy.tools.core.registry import tool

class Toolkit:
    def __init__(
        self,
        name: str = "toolkit",
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        instructions: Optional[str] = None,
        **kwargs
    ):
        """Toolkit base class that bridges to NeoSyntropy registry."""
        self.name = name
        self.tools = tools or []
        self.instructions = instructions
        self.kwargs = kwargs
        self._register_tools()

    def _register_tools(self):
        for func in self.tools:
            # Inspect signature
            sig = inspect.signature(func)
            fields = {}
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                annotation = param.annotation if param.annotation != inspect.Parameter.empty else Any
                default = param.default if param.default != inspect.Parameter.empty else ...
                fields[param_name] = (annotation, default)
            
            # Create Pydantic model for args
            model_name = f"{self.name.capitalize()}_{func.__name__.capitalize()}Args"
            ArgsModel = create_model(model_name, **fields, __base__=BaseModel)

            # Create wrapper function that receives the Pydantic model and unwraps it
            def make_wrapper(f, arg_model):
                def wrapper(args: arg_model) -> Any:
                    return f(**args.model_dump())
                wrapper.__name__ = f.__name__
                wrapper.__doc__ = f.__doc__
                return wrapper

            wrapped_func = make_wrapper(func, ArgsModel)

            # Register with NeoSyntropy
            tool_name = f"{self.name}_{func.__name__}"
            from neosyntropy.tools.core.registry import DEFAULT_REGISTRY
            if tool_name not in DEFAULT_REGISTRY.tools:
                tool(name=tool_name, description=inspect.getdoc(func) or "", args_model=ArgsModel)(wrapped_func)

    def _check_path(
        self,
        relative_path: Union[str, Path],
        base_dir: Union[str, Path],
        restrict_to_base_dir: bool = True,
    ) -> Tuple[bool, Path]:
        """Resolve ``relative_path`` against ``base_dir``.

        When ``restrict_to_base_dir`` is True, reject absolute paths, ``..``
        escapes, and symlink hops that leave the base. Callers treat a False
        first element as deny; the second element is then ``base_dir``.
        """
        base = Path(base_dir).resolve()
        raw = Path(relative_path)
        if restrict_to_base_dir and raw.is_absolute():
            return False, base
        candidate = raw.resolve() if raw.is_absolute() else (base / raw).resolve()
        if restrict_to_base_dir and not candidate.is_relative_to(base):
            return False, base
        return True, candidate

    def get_functions(self) -> List[Callable]:
        return list(self.tools)

    def get_async_functions(self) -> List[Callable]:
        return []
