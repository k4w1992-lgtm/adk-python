# Beta Launch Scope

This document records the scope and decisions for the upcoming ADK Beta Launch.

## Scope Decisions

For the upcoming beta launch, the team has decided on the following scope:

1. **Workflow Path**
   * We will launch the **new workflow path** (`src/google/adk/workflow/_workflow_class.py`). This is the path where `Workflow` inherits from `BaseNode` (**Workflow as BaseNode (Path B)** in [code_paths.md](code_paths.md)).

   | Component | File Path | Description |
   | :--- | :--- | :--- |
   | Orchestrator | `src/google/adk/workflow/_workflow_class.py` | Workflow orchestrator |
   | Execution | `src/google/adk/workflow/_node_runner_class.py` | Node execution loop |
   | Scheduler | `src/google/adk/workflow/_dynamic_node_scheduler.py` | Dynamic node scheduling |
   | Graph | `src/google/adk/workflow/_workflow_graph.py` | Graph definition and validation |
   | Trigger (Shared) | `src/google/adk/workflow/_trigger.py` | Trigger definition |
   | Trigger Processor (Shared) | `src/google/adk/workflow/_trigger_processor.py` | Trigger processing logic |
   | Contract (Shared) | `src/google/adk/workflow/_base_node.py` | BaseNode contract |
   | Context (Shared) | `src/google/adk/agents/context.py` | Execution context |
   | Utils (Shared) | `src/google/adk/workflow/utils/_node_path_utils.py` | Node path utilities |
   | Utils (Shared) | `src/google/adk/workflow/utils/_workflow_hitl_utils.py` | HITL utilities |

2. **Agent Integration**
   * We will launch the **BaseAgent interface using the 1.x wrapper**.
   * This means we will support existing `BaseAgent` implementations (specifically **LlmAgent-1x**, see [code_paths.md](code_paths.md)) by wrapping them as nodes to be executed within the new workflow engine.

   | Component | File Path | Description |
   | :--- | :--- | :--- |
   | **Wrapper** | `src/google/adk/workflow/_v1_llm_agent_wrapper.py` | Wrapper for LlmAgent as node |
   | **Agent** | `src/google/adk/agents/llm_agent_1x.py` | Legacy 1.x LlmAgent |
   | **Flow** | `src/google/adk/flows/llm_flows/base_llm_flow.py` | Core Reason-Act loop flow |
   | **Flow** | `src/google/adk/flows/llm_flows/auto_flow.py` | Flow with sub-agent routing |
   | **Flow** | `src/google/adk/flows/llm_flows/single_flow.py` | Single-turn flow |
   | **Model** | `src/google/adk/models/base_llm.py` | LLM connection interface |
   | **Model** | `src/google/adk/models/google_llm.py` | Gemini implementation |
   | **Schema** | `src/google/adk/models/llm_request.py` | Request schema |
   | **Schema** | `src/google/adk/models/llm_response.py` | Response schema |
## Enabling Beta Features

> [!NOTE]
> To enable the beta features in the command line, set the following environment variables:
> ```bash
> export ADK_ENABLE_NEW_WORKFLOW=true
> export ADK_ENABLE_V1_LLM_AGENT=true
> ```

## Debugging Guidelines

> [!IMPORTANT]
> * **Prioritize this path:** When debugging or investigating issues, prioritize focusing on this code path (Path B for Workflow + LlmAgent-1x via wrapper).
> * **Maintain stability:** Always ensure that this path is working and stable, as it is the target for the beta launch.

## Excluded from Scope

The following paths are NOT included in the initial beta launch scope:
* The **LlmAgent-Node-New** path (Latest Version with Dynamic Nodes, see [code_paths.md](code_paths.md)).
* The **LlmAgent-Workflow-Old** path (LlmAgent with Static Node on old runtime, see [code_paths.md](code_paths.md)).
* The **Workflow as BaseAgent (Path A)** path (`src/google/adk/workflow/_workflow.py`, see [code_paths.md](code_paths.md)).
