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

"""Tests for Workflow class failure handling and cancellation."""

import asyncio

from google.adk.workflow import START
from google.adk.workflow._node_status import NodeStatus
from google.adk.workflow._workflow import WorkflowAgentState
from google.adk.workflow._workflow_class import Workflow
from google.adk.workflow.utils._node_path_utils import join_paths
import pytest

from .workflow_testing_utils import create_parent_invocation_context


@pytest.mark.xfail(
    reason='reimplement after supporting workflow failure handling'
)
@pytest.mark.asyncio
async def test_node_cancellation_on_sibling_failure(
    request: pytest.FixtureRequest,
):
  """Tests that a node is marked as CANCELLED when a sibling node fails."""

  async def slow_node():
    await asyncio.sleep(10)
    yield 'Slow'

  async def fail_node():
    await asyncio.sleep(0.1)
    if False:
      yield
    raise ValueError('Fail')

  agent = Workflow(
      name='test_workflow_cancellation_sibling',
      edges=[
          (START, slow_node),
          (START, fail_node),
      ],
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, agent, resumable=True
  )

  with pytest.raises(ValueError, match='Fail'):
    async for _ in agent.run_async(ctx):
      pass

  # Check persistence
  assert agent.name in ctx.agent_states
  state = WorkflowAgentState.model_validate(ctx.agent_states[agent.name])
  assert state.nodes['fail_node'].status == NodeStatus.FAILED
  assert state.nodes['slow_node'].status == NodeStatus.CANCELLED


@pytest.mark.xfail(
    reason='reimplement after supporting workflow failure handling'
)
@pytest.mark.asyncio
async def test_nested_workflow_cancellation_on_sibling_failure(
    request: pytest.FixtureRequest,
):
  """Tests that a nested workflow and its internal nodes are cancelled."""

  async def inner_slow_node():
    await asyncio.sleep(10)
    yield 'Inner Slow'

  inner_agent = Workflow(
      name='inner_workflow',
      edges=[
          (START, inner_slow_node),
      ],
  )

  async def fail_node():
    await asyncio.sleep(0.1)
    raise ValueError('Fail')

  outer_agent = Workflow(
      name='outer_workflow',
      edges=[
          (START, inner_agent),
          (START, fail_node),
      ],
  )

  ctx = await create_parent_invocation_context(
      request.function.__name__, outer_agent, resumable=True
  )

  with pytest.raises(ValueError, match='Fail'):
    async for _ in outer_agent.run_async(ctx):
      pass

  # Check outer persistence
  assert outer_agent.name in ctx.agent_states
  outer_state = WorkflowAgentState.model_validate(
      ctx.agent_states[outer_agent.name]
  )
  assert outer_state.nodes['fail_node'].status == NodeStatus.FAILED
  assert outer_state.nodes['inner_workflow'].status == NodeStatus.CANCELLED

  # Check inner persistence
  inner_path = join_paths(outer_agent.name, inner_agent.name)
  assert inner_path in ctx.agent_states
  inner_state = WorkflowAgentState.model_validate(ctx.agent_states[inner_path])
  assert inner_state.nodes['inner_slow_node'].status == NodeStatus.CANCELLED
