# Alpha Launch Scope

This document records the scope for the ADK Alpha Launch (Legacy).

## Scope

The alpha launch included the following paths:

1. **Workflow Path (Old)**
   * The **Workflow as BaseAgent (Path A)** path.
   
   | Component | File Path | Description |
   | :--- | :--- | :--- |
   | Orchestrator | `src/google/adk/workflow/_workflow.py` | Workflow inheriting from BaseAgent |
   | Node | `src/google/adk/workflow/_node.py` | Node definition in old runtime |
   | Execution | `src/google/adk/workflow/_node_runner.py` | Node execution loop in old runtime |
   | State (Shared) | `src/google/adk/workflow/_node_state.py` | Node state management |
   | Status (Shared) | `src/google/adk/workflow/_node_status.py` | Node status enum |
   | State | `src/google/adk/workflow/_run_state.py` | Workflow run state |
   | Registry | `src/google/adk/workflow/_dynamic_node_registry.py` | Dynamic node registry |
   | Errors | `src/google/adk/workflow/_errors.py` | Workflow errors |
   | Trigger (Shared) | `src/google/adk/workflow/_trigger.py` | Trigger definition |
   | Trigger Processor (Shared) | `src/google/adk/workflow/_trigger_processor.py` | Trigger processing logic |
   | Contract (Shared) | `src/google/adk/workflow/_base_node.py` | BaseNode contract |
   | Context (Shared) | `src/google/adk/agents/context.py` | Execution context |
   | Utils (Shared) | `src/google/adk/workflow/utils/_node_path_utils.py` | Node path utilities |
   | Utils (Shared) | `src/google/adk/workflow/utils/_workflow_hitl_utils.py` | HITL utilities |

2. **Agent Integration (Static Node)**
   * The **LlmAgent-Workflow-Old** path, using static nodes on the old workflow runtime.

   | Component | File Path | Description |
   | :--- | :--- | :--- |
   | Agent | `src/google/adk/agents/llm_agent_workflow/llm_agent.py` | Static node LlmAgent |
   | Agent | `src/google/adk/agents/llm_agent_workflow/loop_agent.py` | Loop agent |
   | Agent | `src/google/adk/agents/llm_agent_workflow/parallel_agent.py` | Parallel agent |
   | Agent | `src/google/adk/agents/llm_agent_workflow/sequential_agent.py` | Sequential agent |
