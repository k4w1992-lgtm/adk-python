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

"""Shared tool-union conversion utilities."""

from __future__ import annotations

from typing import Any
from typing import cast

from ...models.base_llm import BaseLlm
from ...tools.base_tool import BaseTool
from ...tools.function_tool import FunctionTool


async def _convert_tool_union_to_tools(
    tool_union: Any,
    ctx: Any,
    model: str | BaseLlm,
    multiple_tools: bool = False,
) -> list[BaseTool]:
  """Converts a ToolUnion to a list of BaseTool instances."""
  from ...tools.google_search_tool import GoogleSearchTool
  from ...tools.vertex_ai_search_tool import VertexAiSearchTool

  # Wrap google_search tool with AgentTool if there are multiple tools because
  # the built-in tools cannot be used together with other tools.
  # TODO(b/448114567): Remove once the workaround is no longer needed.
  if multiple_tools and isinstance(tool_union, GoogleSearchTool):
    from ...tools.google_search_agent_tool import create_google_search_agent
    from ...tools.google_search_agent_tool import GoogleSearchAgentTool

    search_tool = cast(GoogleSearchTool, tool_union)
    if search_tool.bypass_multi_tools_limit:
      return [GoogleSearchAgentTool(create_google_search_agent(model))]

  # Replace VertexAiSearchTool with DiscoveryEngineSearchTool if there are
  # multiple tools because the built-in tools cannot be used together with
  # other tools.
  # TODO(b/448114567): Remove once the workaround is no longer needed.
  if multiple_tools and isinstance(tool_union, VertexAiSearchTool):
    from ...tools.discovery_engine_search_tool import DiscoveryEngineSearchTool

    vais_tool = cast(VertexAiSearchTool, tool_union)
    if vais_tool.bypass_multi_tools_limit:
      return [
          DiscoveryEngineSearchTool(
              data_store_id=vais_tool.data_store_id,
              data_store_specs=vais_tool.data_store_specs,
              search_engine_id=vais_tool.search_engine_id,
              filter=vais_tool.filter,
              max_results=vais_tool.max_results,
          )
      ]

  if isinstance(tool_union, BaseTool):
    return [tool_union]
  if callable(tool_union):
    return [FunctionTool(func=tool_union)]

  # At this point, tool_union must be a BaseToolset
  return await tool_union.get_tools_with_prefix(ctx)
