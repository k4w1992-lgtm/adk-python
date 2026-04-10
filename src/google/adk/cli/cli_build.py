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

"""Logic for the `adk build` command."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from typing import Optional

import click
from .utils import build_utils
from .utils import gcp_utils


def build_image(
    agent_folder: str,
    project: Optional[str],
    region: Optional[str],
    repository: str,
    image_name: Optional[str],
    tag: str,
    adk_version: str,
    log_level: str = "INFO",
):
  """Builds an agent image and pushes it to Artifact Registry.

  Args:
    agent_folder: Path to the agent source code.
    project: GCP project ID.
    region: GCP region.
    repository: Artifact Registry repository name.
    image_name: Name of the image. Defaults to agent folder name.
    tag: Image tag.
    adk_version: ADK version to use in the image.
    log_level: Gcloud logging verbosity.
  """
  project = gcp_utils.resolve_project(project)
  env_vars = {}
  # Attempt to read the env variables from .env in the dir (if any).
  env_file = os.path.join(agent_folder, '.env')
  if os.path.exists(env_file):
    from dotenv import dotenv_values

    click.echo(f'Reading environment variables from {env_file}')
    env_vars = dotenv_values(env_file)
    if 'GOOGLE_CLOUD_PROJECT' in env_vars:
      env_project = env_vars.pop('GOOGLE_CLOUD_PROJECT')
      if env_project:
        if project:
          click.secho(
              'Ignoring GOOGLE_CLOUD_PROJECT in .env as `--project` was'
              ' explicitly passed and takes precedence',
              fg='yellow',
          )
        else:
          project = env_project
          click.echo(f'{project=} set by GOOGLE_CLOUD_PROJECT in {env_file}')
    if 'GOOGLE_CLOUD_LOCATION' in env_vars:
      env_region = env_vars.get('GOOGLE_CLOUD_LOCATION')
      if env_region:
        if region:
          click.secho(
              'Ignoring GOOGLE_CLOUD_LOCATION in .env as `--region` was'
              ' explicitly passed and takes precedence',
              fg='yellow',
          )
        else:
          region = env_region
          click.echo(f'{region=} set by GOOGLE_CLOUD_LOCATION in {env_file}')

  app_name = os.path.basename(agent_folder.rstrip("/"))
  image_name = image_name or app_name

  temp_folder = os.path.join(
      tempfile.gettempdir(),
      "adk_build_src",
      datetime.now().strftime("%Y%m%d_%H%M%S"),
  )

  try:
    click.echo(f"Staging build files in {temp_folder}...")
    agent_src_path = os.path.join(temp_folder, "agents", app_name)
    shutil.copytree(agent_folder, agent_src_path)

    requirements_txt_path = os.path.join(agent_src_path, "requirements.txt")
    install_agent_deps = (
        f'RUN pip install -r "/app/agents/{app_name}/requirements.txt"'
        if os.path.exists(requirements_txt_path)
        else "# No requirements.txt found."
    )

    dockerfile_content = build_utils.DOCKERFILE_TEMPLATE.format(
        gcp_project_id=project,
        gcp_region=region,
        app_name=app_name,
        port=8080,  # Default port for container images
        command="api_server",
        install_agent_deps=install_agent_deps,
        service_option=build_utils.get_service_option_by_adk_version(
            adk_version, None, None, None, False
        ),
        trace_to_cloud_option="",
        otel_to_cloud_option="",
        allow_origins_option="",
        adk_version=adk_version,
        host_option="--host=0.0.0.0",
        a2a_option="",
        trigger_sources_option="",
    )

    dockerfile_path = os.path.join(temp_folder, "Dockerfile")
    os.makedirs(temp_folder, exist_ok=True)
    with open(dockerfile_path, "w", encoding="utf-8") as f:
      f.write(dockerfile_content)

    # image URL format: [REGION]-docker.pkg.dev/[PROJECT]/[REPOSITORY]/[IMAGE]:[TAG]
    full_image_url = (
        f"{region}-docker.pkg.dev/{project}/{repository}/{image_name}:{tag}"
    )

    click.secho(f"\nBuilding image: {full_image_url}", bold=True)
    subprocess.run(
        [
            gcp_utils.GCLOUD_CMD,
            "builds",
            "submit",
            "--tag",
            full_image_url,
            "--project",
            project,
            "--verbosity",
            log_level.lower(),
            temp_folder,
        ],
        check=True,
    )
    click.secho("\n✅ Image built and pushed successfully.", fg="green")

  finally:
    if os.path.exists(temp_folder):
      shutil.rmtree(temp_folder)
