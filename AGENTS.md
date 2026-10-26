## Coding conventions

- Use type notation for all functions and methods.
- Use full and readable variable names.
- Do not shorten variable names where normal readable name can be used.
- In docstrings use concise decsription of concept, not detailed code explaination.
- Do not use arbitrary inline comments in code.
- In complex cases where many operations (e.g. reshapes) are stacked, you may add shape notation with inline comment.

For functions and methods taking or returning torch tensors, always specify shapes, ranges and other required properties with this codestyle:

```python
import torch


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    args:
        image: [B, C, H, W], channel_order: RGB, pixel_values: [0...255], dtype: torch.float32
    returns:
        image: [B, C, H, W], channel_order: RGB, pixel_values: [0...1], dtype: torch.float32
    """


def add_tensors(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
    """
    args:
        tensor_a: [B, ...]
        tensor_b: [B, ...]
    returns:
        result: [B, ...]
    """
```

## Development tools

- Never use HEREDOC (`<<EOF`) patterns for file creation or multi-line commands in terminal tool execution. They sometimes hang terminal.
- For code experiments if you need them, use `./dev` directory. You are allowed to make any testing scripts and store any temporary test data there.
