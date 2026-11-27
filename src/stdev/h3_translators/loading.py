import importlib.util
import inspect
import sys
from pathlib import Path

from stdev.h3_translators.base import Translator


def load_translator(translator_path: Path) -> Translator:
    """
    Instantiates the single concrete `Translator` subclass defined in the file.
    """
    spec = importlib.util.spec_from_file_location(translator_path.stem, translator_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[translator_path.stem] = module
    spec.loader.exec_module(module)
    translator_classes = [
        member
        for _, member in inspect.getmembers(module, inspect.isclass)
        if issubclass(member, Translator)
        and member.__module__ == module.__name__
        and not inspect.isabstract(member)
    ]
    if len(translator_classes) != 1:
        raise RuntimeError(f"Expected one Translator in {translator_path}, found {translator_classes}")
    return translator_classes[0]()
