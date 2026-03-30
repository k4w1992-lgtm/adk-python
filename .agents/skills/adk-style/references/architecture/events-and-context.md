# Event Authoring and Context

## Event authoring

Nodes emit events by yielding from `_run_impl()`. The `run()` wrapper
normalizes yields: raw values become `Event(output=value)`, `None` is
skipped, `RequestInput` becomes an interrupt Event. Key fields:

- `event.output` carries the node's result value
- `event.actions.state_delta` carries state mutations
- `event.long_running_tool_ids` signals an interrupt to the caller
- `event.node_info.path` identifies the emitting node in the tree

## Context as node result channel

`ctx` is the source of truth for a node's results. NodeRunner writes
to ctx during execution and returns it to the caller:

- `ctx.output` — the node's result value (at most one per execution)
- `ctx.route` — routing value for conditional edges
- `ctx.interrupt_ids` — accumulated interrupt IDs (read-only for user)

Output and interrupts can coexist — orchestration nodes (Workflow)
allow children to output and interrupt independently. The
orchestrator's `_finalize` decides what to propagate.
