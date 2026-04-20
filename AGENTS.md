# AI Coding Assistant Context

This document provides context for AI coding assistants (Antigravity, Gemini CLI, etc.) to understand the ADK Python project and assist with development.

## ADK Knowledge and Style Guide

For all matters regarding ADK coding style, architecture patterns, and testing best practices, please use the **`adk-style`** skill.
You must read `.agents/skills/adk-style/SKILL.md` to get full instructions when you encounter tasks related to:
- Code style, conventions, and Python idioms
- Architecture patterns (Workflow, Runner, NodeRunner, BaseNode, etc.)
- Testing best practices and rules
- PR reviews and code quality standards

## Project Overview

The Agent Development Kit (ADK) is an open-source, code-first Python toolkit for building, evaluating, and deploying sophisticated AI agents.

### Key Components

- **Agent**: Blueprint defining identity, instructions, and tools.
- **Runner**: Stateless execution engine that orchestrates agent execution.
- **Tool**: Functions/capabilities agents can call.
- **Session**: Conversation state management.
- **Memory**: Long-term recall across sessions.
- **Workflow** (ADK 2.0): Graph-based orchestration of complex, multi-step agent interactions.
- **BaseNode** (ADK 2.0): Contract for all nodes, supporting output streaming and human-in-the-loop steps.
- **Context** (ADK 2.0): Holds execution state and telemetry context mapped 1:1 to nodes.

For details on how the Runner works and the invocation lifecycle, please refer to the `adk-style` skill and the referenced documentation therein.

## Project Architecture

For detailed architecture patterns and component descriptions, please refer to the **`adk-style`** skill at `.agents/skills/adk-style/SKILL.md`.

## Development Setup

The project uses `uv` for package management and Python 3.11+. Please refer to the **`adk-setup`** skill at `.agents/skills/adk-setup/SKILL.md` for detailed instructions.
