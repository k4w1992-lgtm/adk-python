# BaseNode: `@final run()` + `_run_impl()`

Every node follows a two-method pattern:

- `run()` is `@final` — handles lifecycle (context setup, event
  emission, error handling). Never override.
- `_run_impl()` is the extension point — subclasses implement their
  logic here as an async generator that yields outputs.

```python
class MyNode(BaseNode):
    async def _run_impl(self, *, ctx, node_input):
        result = do_work(node_input)
        yield result  # becomes the node's output event
```

**Why this split:** `run()` guarantees consistent lifecycle behavior
(event creation, state flushing, error wrapping) regardless of what
the subclass does. The subclass only thinks about its domain logic.

## Async generator conventions

Nodes yield results via async generators:

```python
async def _run_impl(self, *, ctx, node_input):
    # Yield zero or more intermediate results
    yield intermediate

    # Yield exactly one final output (for most nodes)
    yield final_result
```

- A node that yields nothing produces no output event
- Most nodes yield exactly once (the output)
- Workflow nodes may yield multiple times (one per child completion)
