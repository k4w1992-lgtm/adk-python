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

"""Shared build utilities for ADK CLI."""

from __future__ import annotations

import os
from typing import Final, Optional

from packaging.version import parse

_LOCAL_STORAGE_FLAG_MIN_VERSION: Final[str] = '1.21.0'

DOCKERFILE_TEMPLATE: Final[str] = """
FROM python:3.11-slim
WORKDIR /app

# Create a non-root user
RUN adduser --disabled-password --gecos "" myuser

# Switch to the non-root user
USER myuser

# Set up environment variables - Start
ENV PATH="/home/myuser/.local/bin:$PATH"

ENV GOOGLE_GENAI_USE_VERTEXAI=1
ENV GOOGLE_CLOUD_PROJECT={gcp_project_id}
ENV GOOGLE_CLOUD_LOCATION={gcp_region}

# Set up environment variables - End

# Install ADK - Start
RUN pip install google-adk=={adk_version}
# Install ADK - End

# Copy agent - Start

# Set permission
COPY --chown=myuser:myuser "agents/{app_name}/" "/app/agents/{app_name}/"

# Copy agent - End

# Install Agent Deps - Start
{install_agent_deps}
# Install Agent Deps - End

EXPOSE {port}

CMD adk {command} --port={port} {host_option} {service_option} {trace_to_cloud_option} {otel_to_cloud_option} {allow_origins_option} {a2a_option} {trigger_sources_option} "/app/agents"
"""


def get_service_option_by_adk_version(
    adk_version: str,
    session_uri: Optional[str],
    artifact_uri: Optional[str],
    memory_uri: Optional[str],
    use_local_storage: Optional[bool] = None,
) -> str:
  """Returns service option string based on adk_version."""
  parsed_version = parse(adk_version)
  options: list[str] = []

  if parsed_version >= parse('1.3.0'):
    if session_uri:
      options.append(f'--session_service_uri={session_uri}')
    if artifact_uri:
      options.append(f'--artifact_service_uri={artifact_uri}')
    if memory_uri:
      options.append(f'--memory_service_uri={memory_uri}')
  else:
    if session_uri:
      options.append(f'--session_db_url={session_uri}')
    if parsed_version >= parse('1.2.0') and artifact_uri:
      options.append(f'--artifact_storage_uri={artifact_uri}')

  if use_local_storage is not None and parsed_version >= parse(
      _LOCAL_STORAGE_FLAG_MIN_VERSION
  ):
    # Only valid when session/artifact URIs are unset; otherwise the CLI
    # rejects the combination to avoid confusing precedence.
    if session_uri is None and artifact_uri is None:
      options.append((
          '--use_local_storage'
          if use_local_storage
          else '--no_use_local_storage'
      ))

  return ' '.join(options)
