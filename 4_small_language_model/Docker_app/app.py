import os
import time
from pathlib import Path
from typing import Literal
from uuid import uuid4

import yaml
from fastapi import HTTPException, APIRouter, FastAPI
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from kubernetes.config.config_exception import ConfigException
from pydantic import BaseModel, StrictStr

router = APIRouter()

app = FastAPI(title="AI Job Submission App")


class TaskRequest(BaseModel):
    """The complete and only accepted shape of a task submission."""

    job: Literal["fine-tune", "one-time-prediction"]
    cmd: list[StrictStr]

    class Config:
        # Reject misspelled or unexpected input instead of silently ignoring it.
        extra = "forbid"

JOB_TEMPLATES = {
    "fine-tune": ("fine-tune.yaml", "fine-tune.py"),
    "one-time-prediction": ("one-time-predict.yaml", "one-time-predict.py"),
}


class JobWaitTimeoutError(Exception):
    pass


def _load_kubernetes_config():
    try:
        config.load_incluster_config()
    except ConfigException:
        # This fallback allows local development with the current kubeconfig.
        config.load_kube_config()


def _parse_command(job_type: str, cmd: list[str]) -> list[str]:
    if not isinstance(cmd, list) or not all(isinstance(arg, str) for arg in cmd):
        raise ValueError("cmd must be a JSON array of strings")

    expected_script = JOB_TEMPLATES[job_type][1]
    if (
        len(cmd) < 2
        or cmd[0] != "python3"
        or cmd[1] != expected_script
    ):
        raise ValueError(
            f"Job type {job_type!r} must start with "
            f"['python3', {expected_script!r}]"
        )

    # Never allow model output to choose an arbitrary executable or script.
    return ["python3", expected_script, *cmd[2:]]


def submit_yaml_job(job_type: str, cmd: list[str]) -> tuple[str, str]:
    if job_type not in JOB_TEMPLATES:
        raise ValueError(f"Unsupported Kubernetes job type: {job_type!r}")

    template_filename, _ = JOB_TEMPLATES[job_type]
    template_dir = Path(os.environ.get("JOB_TEMPLATE_DIR", "/app/job_templates"))
    template_path = template_dir / template_filename
    if not template_path.is_file():
        raise RuntimeError(f"Kubernetes Job template not found: {template_path}")

    with template_path.open(encoding="utf-8") as template_file:
        job = yaml.safe_load(template_file)

    if job.get("apiVersion") != "batch/v1" or job.get("kind") != "Job":
        raise ValueError(f"{template_filename} is not a batch/v1 Job template")

    namespace = os.environ.get("KUBERNETES_NAMESPACE", "default")
    name_prefix = "fine-tune" if job_type == "fine-tune" else "one-time-predict"
    job_name = f"{name_prefix}-{uuid4().hex[:8]}"

    job["metadata"]["name"] = job_name
    job["metadata"]["namespace"] = namespace
    job["metadata"].setdefault("labels", {})["submitted-by"] = "ai-agent-app"
    pod_metadata = job["spec"]["template"].setdefault("metadata", {})
    pod_metadata.setdefault("labels", {})["submitted-by"] = "ai-agent-app"
    job["spec"]["template"]["spec"]["containers"][0]["command"] = (
        _parse_command(job_type, cmd)
    )

    _load_kubernetes_config()
    created_job = client.BatchV1Api().create_namespaced_job(
        namespace=namespace,
        body=job,
    )
    return created_job.metadata.name, namespace


def wait_for_job_logs(job_name: str, namespace: str) -> tuple[str, str]:
    try:
        timeout_seconds = int(os.environ.get("JOB_WAIT_TIMEOUT_SECONDS", "1800"))
        poll_seconds = float(os.environ.get("JOB_POLL_INTERVAL_SECONDS", "2"))
    except ValueError as exc:
        raise RuntimeError(
            "Job wait timeout and poll interval must be numeric"
        ) from exc
    if timeout_seconds <= 0 or poll_seconds <= 0:
        raise RuntimeError("Job wait timeout and poll interval must be positive")

    batch_api = client.BatchV1Api()
    core_api = client.CoreV1Api()
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        job = batch_api.read_namespaced_job(
            name=job_name,
            namespace=namespace,
        )
        conditions = {
            condition.type: condition.status
            for condition in (job.status.conditions or [])
        }
        if conditions.get("Complete") == "True":
            final_status = "succeeded"
            break
        if conditions.get("Failed") == "True":
            final_status = "failed"
            break
        time.sleep(poll_seconds)
    else:
        raise JobWaitTimeoutError(
            f"Job {job_name!r} did not finish within {timeout_seconds} seconds"
        )

    pods = core_api.list_namespaced_pod(
        namespace=namespace,
        label_selector=f"job-name={job_name}",
    ).items
    if not pods:
        raise RuntimeError(f"No Pod was found for Job {job_name!r}")

    pods.sort(key=lambda pod: pod.metadata.creation_timestamp)
    log_sections = []
    for pod in pods:
        pod_name = pod.metadata.name
        logs = core_api.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
        )
        log_sections.append(f"===== {pod_name} =====\n{logs}")

    return final_status, "\n\n".join(log_sections)


@router.post("/tasks")
def submit_task(request: TaskRequest):
    try:
        job_id, namespace = submit_yaml_job(request.job, request.cmd)
        status, logs = wait_for_job_logs(job_id, namespace)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except JobWaitTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except (ApiException, RuntimeError) as exc:
        raise HTTPException(
            status_code=503, detail=f"Kubernetes operation failed: {exc}"
        ) from exc

    return {
        "status": status,
        "job_id": job_id,
        "logs": logs,
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# Routers must be included after their routes have been declared.
app.include_router(router)
