import functools
from typing import Callable
from .dataset import Input, Output

def transform(*, out: Output, **inputs):
    """
    Decorator for knowledge transformation pipelines.
    
    Example:
        @transform(
            out=Output("enriched_knowledge"),
            raw_data=Input("raw_source")
        )
        def compute_enrichment(raw_data: Input, out: Output):
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Pass the configured Input and Output instances to the compute function
            kwargs.update(inputs)
            kwargs['out'] = out
            return func(*args, **kwargs)
        
        # Attach metadata for potential pipeline registration
        wrapper.__is_transform__ = True
        return wrapper
    return decorator
