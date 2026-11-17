"""
Entry point of the STDev development node pack. Each VAE upscaler lives in its own module under
src/stdev/comfy_nodes/<vae>/nodes.py; register new ones in NODE_MODULES.
"""

import importlib

NODE_MODULES = ("stdev.comfy_nodes.wan_2_1.nodes",)

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}
for module_name in NODE_MODULES:
    module = importlib.import_module(module_name)
    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
