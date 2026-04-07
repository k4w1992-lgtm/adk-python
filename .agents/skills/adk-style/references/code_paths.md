# Code Paths and Key Files

This document lists the key files along the different code paths for `LlmAgent` and `Workflow` in the ADK project. This is intended to help AI and humans understand the context when tracing the code.

## Agent Code Paths

There are 3 code paths for `LlmAgent` or `BaseAgent` interface:

1. **Legacy 1.x Version** (Name: `LlmAgent-1x`)
   * **Path:** `src/google/adk/agents/llm_agent_1x.py`
   * **Context:** This is the version currently in `main`. It uses legacy flows and wrappers.

2. **LlmAgent with Static Node (Old Workflow Runtime)** (Name: `LlmAgent-Workflow-Old`)
   * **Path:** `src/google/adk/agents/llm_agent_workflow`
   * **Context:** This folder hosts the `LlmAgent` implementation using static nodes, based on the OLD workflow runtime.

3. **Latest Version with Dynamic Nodes** (Name: `LlmAgent-Node-New`)
   * **Path:** `src/google/adk/agents/llm_agent_node`
   * **Context:** This folder hosts the latest version using dynamic nodes.

## Workflow Code Paths

There are 2 modern workflow code paths (sharing `Context` and `BaseNode` definitions):

1. **Workflow as BaseAgent (Path A)**
   * **Path:** `src/google/adk/workflow/_workflow.py`
   * **Context:** Defines `Workflow` inheriting from `BaseAgent` and `Node`. It implements `NodeRunner` and `Workflow` in this specific runtime.

2. **Workflow as BaseNode (Path B) (New Path)**
   * **Path:** `src/google/adk/workflow/_workflow_class.py`
   * **Context:** Defines `Workflow` inheriting from `BaseNode`. It implements `NodeRunner` and `Workflow` in a runtime aligned with the new `BaseNode` contract.

Both paths share:
* `src/google/adk/workflow/_base_node.py` (BaseNode definition)
* `src/google/adk/agents/context.py` (Context definition)
