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

"""Tests for adk build command."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import click
from click.testing import CliRunner
from google.adk.cli import cli_build
from google.adk.cli import cli_tools_click
import pytest


@pytest.fixture(autouse=True)
def _mute_click(monkeypatch: pytest.MonkeyPatch) -> None:
  """Suppress click output during tests."""
  monkeypatch.setattr(click, "echo", lambda *a, **k: None)


def test_cli_build_invokes_build_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """adk build image should forward arguments to cli_build.build_image."""
  mock_run = mock.Mock()
  monkeypatch.setattr(cli_tools_click.cli_build, "build_image", mock_run)

  agent_dir = tmp_path / "my_agent"
  agent_dir.mkdir()

  runner = CliRunner()
  result = runner.invoke(
      cli_tools_click.main,
      [
          "build",
          "image",
          "--project",
          "my-project",
          "--region",
          "us-central1",
          "--repository",
          "my-repo",
          "--image_name",
          "my-image",
          "--tag",
          "v1",
          str(agent_dir),
      ],
  )

  assert result.exit_code == 0
  mock_run.assert_called_once()
  kwargs = mock_run.call_args.kwargs
  assert kwargs["project"] == "my-project"
  assert kwargs["region"] == "us-central1"
  assert kwargs["repository"] == "my-repo"
  assert kwargs["image_name"] == "my-image"
  assert kwargs["tag"] == "v1"
  assert kwargs["agent_folder"] == str(agent_dir)


def test_cli_build_directory_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Exception from build_image should be caught and surfaced."""

  def _boom(*args, **kwargs):
    raise RuntimeError("build error")

  monkeypatch.setattr(cli_tools_click.cli_build, "build_image", _boom)

  agent_dir = tmp_path / "my_agent_fail"
  agent_dir.mkdir()

  runner = CliRunner()
  result = runner.invoke(
      cli_tools_click.main,
      ["build", "image", "--repository", "repo", str(agent_dir)],
  )

  assert result.exit_code == 0
  assert "Build failed: build error" in result.output


def test_build_image_success(tmp_path, monkeypatch):
  """build_image should construct and run the correct gcloud command."""
  mock_subprocess = mock.Mock()
  monkeypatch.setattr(cli_build.subprocess, "run", mock_subprocess)

  monkeypatch.setattr(
      cli_build.gcp_utils, "resolve_project", lambda x: "test-project"
  )

  agent_dir = tmp_path / "my_agent"
  agent_dir.mkdir()
  (agent_dir / "agent.py").touch()

  cli_build.build_image(
      agent_folder=str(agent_dir),
      project="test-project",
      region="us-east1",
      repository="test-repo",
      image_name="test-image",
      tag="latest",
      adk_version="1.3.0",
  )

  mock_subprocess.assert_called()
  args = mock_subprocess.call_args[0][0]
  assert "builds" in args
  assert "submit" in args
  assert "--tag" in args
  assert (
      "us-east1-docker.pkg.dev/test-project/test-repo/test-image:latest" in args
  )
