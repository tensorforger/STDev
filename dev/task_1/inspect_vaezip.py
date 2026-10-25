import inspect

import torch
import vaezip

print(inspect.signature(vaezip.save_tensor))
print(inspect.getdoc(vaezip.save_tensor))
print(inspect.signature(vaezip.load_tensor))
print(inspect.getdoc(vaezip.load_tensor))
print(torch.__version__)
