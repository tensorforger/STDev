import inspect

import vaezip

for function_name in ("encode", "save"):
    function = getattr(vaezip, function_name)
    print(function_name, inspect.signature(function))
    print(inspect.getdoc(function))
    print()
