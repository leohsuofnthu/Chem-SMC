"""
Convenience imports for experiment modules.
Lazy imports to avoid requiring all dependencies at package import time.
"""

__all__ = [
    "evaluate",
    "plots",
    "smiley_generate",
    "utils",
]


def __getattr__(name: str):
    """Lazy import of modules to avoid requiring all dependencies at package import time."""
    if name == "evaluate":
        from . import evaluate
        return evaluate
    elif name == "plots":
        from . import plots
        return plots
    elif name == "smiley_generate":
        from . import smiley_generate
        return smiley_generate
    elif name == "utils":
        from . import utils
        return utils
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
