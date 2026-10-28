"""
Shared helpers for dev scripts: load a run file as a module with constant overrides and redirected
output directories.
"""

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
UPSCALERS_DIRECTORY = REPOSITORY_ROOT / "src" / "stdev" / "upscalers"


def resolve_run_file(run: str) -> Path:
    if run.isdigit():
        return UPSCALERS_DIRECTORY / f"run_{int(run):04d}.py"
    return Path(run).resolve()


def load_run_module(run_file: Path, overrides: list[str]) -> ModuleType:
    """
    Executes the run file as a module, then applies NAME=EXPR overrides (EXPR evaluated in the
    module namespace, so `torch.float16` or `(128, 256)` work).
    """
    module_name = f"dev_{run_file.stem}"
    spec = importlib.util.spec_from_file_location(module_name, run_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    for override in overrides:
        name, expression = override.split("=", 1)
        if not hasattr(module, name):
            raise AttributeError(f"{run_file.name} has no constant {name}")
        setattr(module, name, eval(expression, module.__dict__))
    return module


def instantiate_upscaler(module: ModuleType):
    from stdev.upscalers.base import Upscaler

    classes = [
        member
        for _, member in inspect.getmembers(module, inspect.isclass)
        if issubclass(member, Upscaler)
        and member.__module__ == module.__name__
        and not inspect.isabstract(member)
    ]
    return classes[0]()
