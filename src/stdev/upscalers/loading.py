import importlib.util
import inspect
import sys
from pathlib import Path

from stdev.upscalers.base import Upscaler


def load_upscaler(upscaler_path: Path) -> Upscaler:
    """
    Instantiates the single concrete `Upscaler` subclass defined in the file.
    """
    spec = importlib.util.spec_from_file_location(upscaler_path.stem, upscaler_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[upscaler_path.stem] = module
    spec.loader.exec_module(module)
    upscaler_classes = [
        member
        for _, member in inspect.getmembers(module, inspect.isclass)
        if issubclass(member, Upscaler)
        and member.__module__ == module.__name__
        and not inspect.isabstract(member)
    ]
    if len(upscaler_classes) != 1:
        raise RuntimeError(f"Expected one Upscaler in {upscaler_path}, found {upscaler_classes}")
    return upscaler_classes[0]()
