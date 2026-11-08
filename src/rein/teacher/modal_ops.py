import json
import shutil
import subprocess

from rein.env import load_env, optional
from rein.teacher.constants import ENV_TEACHER_APP

APP_NAME_FIELD = "App Name"
CONTAINER_ID_FIELD = "Container ID"
CLI_TIMEOUT_S = 120

TEACHER_APP = optional(load_env(), ENV_TEACHER_APP)


def stop_app_containers(app_name: str | None) -> list[str]:
    """Stops every running container of a Modal app so nothing bills while idle. Returns the stopped ids. Without an
    app name (an endpoint that is not a Modal app, or one left running on purpose) nothing is stopped."""
    if app_name is None:
        return []
    modal = shutil.which("modal")
    if modal is None:
        raise RuntimeError("the modal CLI is not on PATH")
    listing = subprocess.run(
        [modal, "container", "list", "--json"], capture_output=True, text=True, check=True, timeout=CLI_TIMEOUT_S
    )
    stopped = []
    for container in json.loads(listing.stdout or "[]"):
        if container.get(APP_NAME_FIELD) != app_name:
            continue
        container_id = container[CONTAINER_ID_FIELD]
        subprocess.run(
            [modal, "container", "stop", container_id], check=True, timeout=CLI_TIMEOUT_S, capture_output=True
        )
        stopped.append(container_id)
    return stopped
