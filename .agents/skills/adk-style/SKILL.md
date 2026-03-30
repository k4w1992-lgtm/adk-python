---
name: adk-style
description: ADK development style guide — architecture patterns, testing best practices, Python idioms, and codebase conventions. Use when writing code, tests, or reviewing PRs for the ADK project. Triggers on "write tests", "best practice", "code style", "how should I", "convention", "pattern", "review code".
---

# ADK Style Guide

This skill has three reference documents:

- [Development Conventions](references/development.md) — public API vs internal methods, comments, file organization, imports, Pydantic patterns, formatting
- Architecture Reference (references/architecture/) — split by topic:
  - [BaseNode](references/architecture/base-node.md) — `@final run()` + `_run_impl()`, async generator conventions
  - [Runner](references/architecture/runner.md) — Runner vs NodeRunner vs Workflow separation
  - [Events and Context](references/architecture/events-and-context.md) — event authoring, context as result channel
  - [Checkpoint and Resume](references/architecture/checkpoint-resume.md) — HITL lifecycle, `rerun_on_resume`, `execution_id`
- [Testing Guide](references/testing.md) — core principles, 9 rules for writing ADK tests, test structure template
