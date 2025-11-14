"""
Convenience imports for experiment modules.
Lazy imports to avoid requiring all dependencies at package import time.
"""
import importlib

__all__ = [
    "evaluate",
    "plots",
    "smiley_generate",
    "utils",
]

# Cache for imported modules to avoid recursion
_import_cache = {}


def __getattr__(name: str):
    """Lazy import of modules to avoid requiring all dependencies at package import time."""
    if name in _import_cache:
        return _import_cache[name]
    
    # Use importlib to avoid recursion during import
    if name == "evaluate":
        module = importlib.import_module(".evaluate", __name__)
        _import_cache[name] = module
        return module
    elif name == "plots":
        module = importlib.import_module(".plots", __name__)
        _import_cache[name] = module
        return module
    elif name == "smiley_generate":
        module = importlib.import_module(".smiley_generate", __name__)
        _import_cache[name] = module
        return module
    elif name == "utils":
        module = importlib.import_module(".utils", __name__)
        _import_cache[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
