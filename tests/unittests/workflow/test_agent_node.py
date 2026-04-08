# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for AgentNode."""

from __future__ import annotations

from typing import AsyncGenerator

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.sessions.session import Session
from google.adk.workflow._agent_node import AgentNode
import pytest


class MockAgent(BaseAgent):
  """A mock agent that yields predefined events."""

  async def _run_async_impl(
      self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    yield Event(author=self.name, message="hello")
    yield Event(author="sub_agent", message="world")


@pytest.mark.asyncio
async def test_agent_node_init():
  """Tests that AgentNode initializes correctly and inherits name."""
  agent = MockAgent(name="mock_agent")
  node = AgentNode(agent=agent)
  assert node.name == "mock_agent"


@pytest.mark.asyncio
async def test_agent_node_model_copy():
  """Tests that model_copy propagates name updates to the agent."""
  agent = MockAgent(name="mock_agent")
  node = AgentNode(agent=agent)
  copied = node.model_copy(update={"name": "new_name"})
  assert copied.name == "new_name"
  assert copied.agent.name == "new_name"


@pytest.mark.asyncio
async def test_agent_node_run():
  """Tests that AgentNode runs the agent and preserves event authors."""
  agent = MockAgent(name="mock_agent")
  node = AgentNode(agent=agent)

  # Setup minimal context
  session = Session(app_name="test", user_id="user", id="session")
  session_service = InMemorySessionService()
  ic = InvocationContext(
      invocation_id="inv",
      session=session,
      session_service=session_service,
  )
  ctx = Context(ic, node_path="wf")

  events = []
  async for event in node.run(ctx=ctx, node_input=None):
    events.append(event)

  assert len(events) == 2

  # First event from mock_agent
  assert events[0].author == "mock_agent"
  assert events[0].node_info.path == "wf"

  # Second event from sub_agent
  assert events[1].author == "sub_agent"
  # Path should not be set by AgentNode for sub_agent if author doesn't match agent name
  assert not events[1].node_info.path

  # Also check if ctx.event_author was updated to preserve author for NodeRunner
  assert ctx.event_author == "sub_agent"
