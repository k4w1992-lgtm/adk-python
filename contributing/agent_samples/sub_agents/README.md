# ADK Agent Sub-Agents Sample

## Overview

This sample demonstrates how to create a hierarchical agent setup using sub-agents in the **ADK** framework.

It defines a root `Agent` that coordinates two sub-agents: `random_number_agent` and `is_even_agent`. Each sub-agent is equipped with its own specific Python function tool. The root agent delegates tasks to these sub-agents based on the user's prompt. This sample illustrates how to modularize capabilities into separate agents instead of combining all tools on a single agent.

## Sample Prompts

- `Give me a random number.`
- `Give me a random number up to 50, and tell me if it's even.`
- `Is 44 even?`

## Graph

```text
Agent (name="sub_agents")
├── Agent (name="random_number_agent")
│   └── Tool (name="generate_random_number")
└── Agent (name="is_even_agent")
    └── Tool (name="is_even")
```

## How To

1. Define the specific tools for each sub-agent:

   ```python
   import random

   def generate_random_number(max_value: int = 100) -> int:
       """Generates a random integer between 0 and max_value (inclusive). ..."""
       return random.randint(0, max_value)

   def is_even(number: int) -> bool:
       """Checks if a given number is even. ..."""
       return number % 2 == 0
   ```

1. Register tools to their respective sub-agents:

   ```python
   from google.adk.agents import Agent

   random_number_agent = Agent(
       name="random_number_agent",
       description="An agent that can generate a random number.",
       tools=[generate_random_number],
   )

   is_even_agent = Agent(
       name="is_even_agent",
       description="An agent that can check if a given number is even.",
       tools=[is_even],
   )
   ```

1. Add the sub-agents to the root agent's `sub_agents` list:

   ```python
   root_agent = Agent(
       name="sub_agents",
       sub_agents=[random_number_agent, is_even_agent],
   )
   ```
