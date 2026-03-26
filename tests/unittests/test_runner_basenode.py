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

"""Tests for Runner driving a BaseNode root.

Verifies that Runner can accept a BaseNode (not just BaseAgent),
drive it through NodeRunner, persist events to session, and
yield them to the caller.
"""

from typing import Any
from typing import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.workflow._base_node import BaseNode
from google.adk.workflow._base_node import START
from google.adk.workflow._workflow_class import Workflow
from google.genai import types
import pytest

# --- Shared helper nodes ---


class _EchoNode(BaseNode):

  async def _run_impl(
      self, *, ctx: Context, node_input: Any
  ) -> AsyncGenerator[Any, None]:
    text = node_input.parts[0].text if node_input else 'empty'
    yield f'Echo: {text}'


# --- Helpers ---


async def _run(node, message='hello'):
  """Run a node through Runner, return collected events."""
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=node, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')
  msg = types.Content(parts=[types.Part(text=message)], role='user')
  events = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg
  ):
    events.append(event)
  return events, ss, session


# --- Tests ---


@pytest.mark.asyncio
async def test_node_output_returned_to_caller():
  """Runner yields the node's output event to the caller."""
  events, _, _ = await _run(_EchoNode(name='echo'), message='hi')

  outputs = [e.output for e in events if e.output is not None]
  assert outputs == ['Echo: hi']


@pytest.mark.asyncio
async def test_events_persisted_to_session():
  """Non-partial events are persisted to the session."""
  _, ss, session = await _run(_EchoNode(name='echo'), message='hi')

  updated = await ss.get_session(
      app_name='test', user_id='u', session_id=session.id
  )
  session_outputs = [e.output for e in updated.events if e.output is not None]
  assert 'Echo: hi' in session_outputs


@pytest.mark.asyncio
async def test_non_output_events_also_yielded():
  """Runner yields intermediate events (e.g. state), not just output."""

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield Event(state={'step': 'processing'})
      yield 'final_result'

  events, _, _ = await _run(_Node(name='steps'))

  state_events = [e for e in events if e.actions and e.actions.state_delta]
  assert len(state_events) >= 1
  assert [e.output for e in events if e.output is not None] == ['final_result']


@pytest.mark.asyncio
async def test_node_error_completes_without_events():
  """A node that raises completes the invocation with no output."""

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      raise RuntimeError('node failure')
      yield  # pylint: disable=unreachable

  # TODO: Propagate node errors to the caller. Currently the error
  # is swallowed by the background task.
  events, _, _ = await _run(_Node(name='error'))

  assert [e.output for e in events if e.output is not None] == []


@pytest.mark.asyncio
async def test_event_author_defaults_to_node_name():
  """Events are attributed to the node's name by default."""
  events, _, _ = await _run(_EchoNode(name='my_node'), message='hi')

  output_events = [e for e in events if e.output is not None]
  assert output_events[0].author == 'my_node'


# --- Resume tests ---


@pytest.mark.asyncio
async def test_standalone_node_resume():
  """A standalone node resumes with resume_inputs from function response."""

  class _Node(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        yield f'result: {ctx.resume_inputs["fc-1"]["value"]}'
        return
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='get_input', args={}, id='fc-1'
                      )
                  )
              ]
          ),
          long_running_tool_ids={'fc-1'},
      )

  node = _Node(name='standalone')
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=node, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: node interrupts
  msg1 = types.Content(parts=[types.Part(text='go')], role='user')
  events1: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  assert any(e.long_running_tool_ids for e in events1)

  # Run 2: resume with function response
  msg2 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='get_input',
                  id='fc-1',
                  response={'value': 42},
              )
          )
      ],
      role='user',
  )
  events2: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg2
  ):
    events2.append(event)

  outputs = [e.output for e in events2 if e.output is not None]
  assert 'result: 42' in outputs


@pytest.mark.xfail(reason='Workflow resume not yet implemented.')
@pytest.mark.asyncio
async def test_workflow_resume_after_interrupt():
  """Workflow node resumes from HITL interrupt and triggers downstream."""

  class _InterruptNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        yield f'approved: {ctx.resume_inputs["fc-1"]["approved"]}'
        return
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='ask_approval', args={}, id='fc-1'
                      )
                  )
              ]
          ),
          long_running_tool_ids={'fc-1'},
      )

  class _AfterNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      yield f'done: {node_input}'

  wf = Workflow(
      name='wf',
      edges=[(START, _InterruptNode(name='ask'), _AfterNode(name='after'))],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Given a workflow that interrupts on first run
  msg1 = types.Content(parts=[types.Part(text='start')], role='user')
  events1: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  assert any(e.long_running_tool_ids for e in events1)

  # When the user responds with a function response
  # (invocation_id auto-resolved from FR matching the FC event)
  msg2 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='ask_approval',
                  id='fc-1',
                  response={'approved': True},
              )
          )
      ],
      role='user',
  )
  events2: list[Event] = []
  async for event in runner.run_async(
      user_id='u',
      session_id=session.id,
      new_message=msg2,
  ):
    events2.append(event)

  # Then the interrupted node resumes and downstream runs
  outputs = [e.output for e in events2 if e.output is not None]
  assert 'approved: True' in outputs
  assert any('done' in str(o) for o in outputs)


@pytest.mark.asyncio
async def test_resume_raises_on_unmatched_fr():
  """Runner raises when function response has no matching FC in session."""
  ss = InMemorySessionService()
  runner = Runner(
      app_name='test',
      node=_EchoNode(name='echo'),
      session_service=ss,
  )
  session = await ss.create_session(app_name='test', user_id='u')

  msg = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='unknown',
                  id='no-such-fc',
                  response={'x': 1},
              )
          )
      ],
      role='user',
  )

  with pytest.raises(ValueError, match='Function call not found'):
    async for _ in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg
    ):
      pass


@pytest.mark.asyncio
async def test_resume_raises_on_multi_invocation_fr():
  """Runner raises when FRs resolve to different invocations."""
  call_count = [0]

  class _InterruptNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      call_count[0] += 1
      fc_id = f'fc-{call_count[0]}'
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='tool', args={}, id=fc_id
                      )
                  )
              ]
          ),
          long_running_tool_ids={fc_id},
      )

  wf = Workflow(
      name='wf',
      edges=[(START, _InterruptNode(name='ask'))],
  )
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=wf, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: interrupts with fc-1
  async for _ in runner.run_async(
      user_id='u',
      session_id=session.id,
      new_message=types.Content(parts=[types.Part(text='go')], role='user'),
  ):
    pass

  # Run 2: interrupts with fc-2 (different invocation)
  async for _ in runner.run_async(
      user_id='u',
      session_id=session.id,
      new_message=types.Content(
          parts=[types.Part(text='go again')], role='user'
      ),
  ):
    pass

  # Run 3: send FRs for both fc-1 and fc-2 (different invocations)
  msg3 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='tool', id='fc-1', response={'r': 1}
              )
          ),
          types.Part(
              function_response=types.FunctionResponse(
                  name='tool', id='fc-2', response={'r': 2}
              )
          ),
      ],
      role='user',
  )

  with pytest.raises(ValueError, match='resolve to multiple invocations'):
    async for _ in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg3
    ):
      pass


# --- yield_user_message tests ---


@pytest.mark.asyncio
async def test_yield_user_message_true():
  """When yield_user_message=True, user event is yielded before node events."""
  events, _, _ = await _run(_EchoNode(name='echo'), message='hi')

  # Default (False) — no user event in output
  user_events = [e for e in events if e.author == 'user']
  assert user_events == []

  # Now with yield_user_message=True
  ss = InMemorySessionService()
  runner = Runner(
      app_name='test', node=_EchoNode(name='echo'), session_service=ss
  )
  session = await ss.create_session(app_name='test', user_id='u')
  msg = types.Content(parts=[types.Part(text='hi')], role='user')

  events_with_user: list[Event] = []
  async for event in runner.run_async(
      user_id='u',
      session_id=session.id,
      new_message=msg,
      yield_user_message=True,
  ):
    events_with_user.append(event)

  user_events = [e for e in events_with_user if e.author == 'user']
  assert len(user_events) == 1
  assert user_events[0].content.parts[0].text == 'hi'
  # User event should be first
  assert events_with_user[0].author == 'user'


@pytest.mark.asyncio
async def test_yield_user_message_false_by_default():
  """By default, user event is not yielded to the caller."""
  events, _, _ = await _run(_EchoNode(name='echo'), message='hi')

  user_events = [e for e in events if e.author == 'user']
  assert user_events == []


# --- Edge case tests ---


@pytest.mark.asyncio
async def test_find_original_user_content_with_multiple_messages():
  """Runner finds original text message even with multiple user events."""

  class _InterruptNode(BaseNode):

    async def _run_impl(
        self, *, ctx: Context, node_input: Any
    ) -> AsyncGenerator[Any, None]:
      if ctx.resume_inputs and 'fc-1' in ctx.resume_inputs:
        # On resume, verify we got the original text, not the FR
        text = (
            node_input.parts[0].text
            if node_input and hasattr(node_input, 'parts')
            else str(node_input)
        )
        yield f'original:{text}'
        return
      yield Event(
          content=types.Content(
              parts=[
                  types.Part(
                      function_call=types.FunctionCall(
                          name='tool', args={}, id='fc-1'
                      )
                  )
              ]
          ),
          long_running_tool_ids={'fc-1'},
      )

  node = _InterruptNode(name='node')
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=node, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1: send text
  msg1 = types.Content(
      parts=[types.Part(text='my original input')], role='user'
  )
  async for _ in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    pass

  # Run 2: resume with FR
  msg2 = types.Content(
      parts=[
          types.Part(
              function_response=types.FunctionResponse(
                  name='tool', id='fc-1', response={'v': 1}
              )
          )
      ],
      role='user',
  )
  events2: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg2
  ):
    events2.append(event)

  outputs = [e.output for e in events2 if e.output is not None]
  assert 'original:my original input' in outputs


@pytest.mark.asyncio
async def test_mixed_fr_and_text_raises():
  """Message with both function responses and text is rejected."""
  ss = InMemorySessionService()
  runner = Runner(
      app_name='test', node=_EchoNode(name='echo'), session_service=ss
  )
  session = await ss.create_session(app_name='test', user_id='u')

  msg = types.Content(
      parts=[
          types.Part(text='some text'),
          types.Part(
              function_response=types.FunctionResponse(
                  name='tool', id='fc-1', response={'v': 1}
              )
          ),
      ],
      role='user',
  )

  with pytest.raises(ValueError, match='cannot contain both'):
    async for _ in runner.run_async(
        user_id='u', session_id=session.id, new_message=msg
    ):
      pass


@pytest.mark.asyncio
async def test_plain_text_message_does_not_trigger_resume():
  """Sending plain text (no FR) starts fresh, does not enter resume path."""
  node = _EchoNode(name='echo')
  ss = InMemorySessionService()
  runner = Runner(app_name='test', node=node, session_service=ss)
  session = await ss.create_session(app_name='test', user_id='u')

  # Run 1
  msg1 = types.Content(parts=[types.Part(text='first')], role='user')
  events1: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg1
  ):
    events1.append(event)

  # Run 2: plain text, not FR — should start fresh
  msg2 = types.Content(parts=[types.Part(text='second')], role='user')
  events2: list[Event] = []
  async for event in runner.run_async(
      user_id='u', session_id=session.id, new_message=msg2
  ):
    events2.append(event)

  outputs = [e.output for e in events2 if e.output is not None]
  assert outputs == ['Echo: second']
