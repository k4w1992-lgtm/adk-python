---
name: git-ops
description: Use for any git operation (commit, push, pull, rebase, branch, PR, cherry-pick, etc.). Provides commit message format and conventions.
---

# Git Operations for adk-python

## Commit Message Format

Use **Conventional Commits**:

```
<type>(<scope>): <description>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, no code change
- `refactor`: Code restructure without behavior change
- `perf`: Performance improvement
- `test`: Adding/updating tests
- `chore`: Build, config, dependencies
- `ci`: CI/CD changes

### Description Phrasing

**CRITICAL**: The subject line must answer **why**, not just **what**.
A reviewer reading only the subject should understand the motivation.

- **State the outcome**, not the mechanics:
  - Good: `Fix race condition when two agents write to same session`
  - Bad: `Update session.py to add lock`
- **Name the capability added**, not the implementation:
  - Good: `Support parallel tool execution in workflows`
  - Bad: `Add asyncio.gather call in execute_tools_node`
- **For refactors, state the reason**, not just the action:
  - Good: `Make graph public for dev UI serialization`
  - Bad: `Make graph a public field on new Workflow`
- **For bug fixes, state what was broken**:
  - Good: `Prevent duplicate events when resuming HITL`
  - Bad: `Check interrupt_id before appending`

Self-check before committing: read your subject line and ask
"does this tell me *why* someone made this change?" If it only
describes *what* changed, rewrite it.

### Rules

1. **Imperative mood** - "Add feature" not "Added feature".
2. **Capitalize** first letter of description (for release-please changelog).
3. **No period** at end of subject line.
4. **50 char limit** on subject line when possible, max 72.
5. **Use body for context** - Add a blank line then explain *why*,
   not *how*, when the subject alone isn't enough.

### Examples

```
feat(agents): Support App pattern with lifecycle plugins
fix(sessions): Prevent memory leak on concurrent session cleanup
refactor(tools): Unify env var checks across tool implementations
docs: Add contributing guide for first-time contributors
```
