## Context handling

This repository can be cloned to different machines, training environments with several GPUs are not persistent, everything that is not tracked by git or uploaded to R2 storage would be wiped out after training or research. To track the context and knowledge we are using:

- **CONTEXT.md** - long-term context of the project, only short paragraphs, high level findings.
- **./reports** - more details, tables, numbers, script entry points.
- **./autoresearch_* directories** - two files in each: **instruction.md** - the research setup and questions, **experiment_logs.md** - detailed context of research process.

You can freely add information in reports and autoresearch logs (when you are doing autoresearch). Add a block to CONTEXT.md only when user asks you.

## Development tools

- For code experiments if you need them, use `./dev` directory. You are allowed to make any testing scripts and store any temporary test data there.
- Git: Do not make separate git branches and commit anything until explicitly asked by user.

## Long training run control

If you are asked to control the long training run follow these rules:

- Run the trainig script in the background terminal
- Keep only the error watcher (that would notify you about exceptions and script exit).
- NEVER poll the run constantly. Checking run's status manually every n minutes in a loop is forbidden. This is almost always wasteful.
- You can only add a single timer-based notifiaction if there is a reason to do this. For example, check the progress once after 10 minutes.
- If you are waiting for some condition to be met, make a script that would check it and notify you.

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