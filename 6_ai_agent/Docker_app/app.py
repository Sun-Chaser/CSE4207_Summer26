import json
import os
from pathlib import Path
from uuid import uuid4

import httpx
import yaml
from fastapi import HTTPException, APIRouter, FastAPI
from kubernetes import client, config
from kubernetes.client.exceptions import ApiException
from kubernetes.config.config_exception import ConfigException
from pydantic import BaseModel

router = APIRouter()

app = FastAPI(title="AI Job Submission App")

AI_AGENT_API_URL = os.environ.get(
    "AI_AGENT_API_URL", "http://host.minikube.internal:8000"
).rstrip("/")

class TaskRequest(BaseModel):
    prompt: str

JOB_TEMPLATES = {
    "fine-tune": ("fine-tune.yaml", "fine-tune.py"),
    "one-time-prediction": ("one-time-predict.yaml", "one-time-predict.py"),
}


def _load_kubernetes_config():
    try:
        config.load_incluster_config()
    except ConfigException:
        # This fallback allows local development with the current kubeconfig.
        config.load_kube_config()


def _parse_command(job_type: str, cmd: str | list[str]) -> list[str]:
    if isinstance(cmd, str):
        try:
            cmd = json.loads(cmd)
        except json.JSONDecodeError as exc:
            raise ValueError("The generated command is not valid JSON") from exc

    if not isinstance(cmd, list) or not all(isinstance(arg, str) for arg in cmd):
        raise ValueError("The generated command must be a list of strings")

    expected_script = JOB_TEMPLATES[job_type][1]
    allowed_scripts = {expected_script}
    if job_type == "one-time-prediction":
        # The agent's training examples contain this older misspelling.
        allowed_scripts.add("one-time-predic.py")

    if (
        len(cmd) < 2
        or cmd[0] not in {"python", "python3"}
        or Path(cmd[1]).name not in allowed_scripts
    ):
        raise ValueError(f"Job type {job_type!r} must run {expected_script!r}")

    # Never allow model output to choose an arbitrary executable or script.
    return ["python3", expected_script, *cmd[2:]]


def submit_yaml_job(job_type: str, cmd: str | list[str]) -> str:
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
    return created_job.metadata.name

@router.post("/tasks")
async def submit_task(request: TaskRequest):
    async with httpx.AsyncClient(timeout=300.0) as client:
        try:
            response = await client.post(
                f"{AI_AGENT_API_URL}/parse",
                params={"prompt": request.prompt},
            )
            response.raise_for_status()
        except httpx.RequestError as exc:
            raise HTTPException(status_code=503, detail=f"Service unavailable: {exc}")
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail="AI agent API returned an error",
            )

    try:
        data = response.json()
        # Docker_api currently returns the generated JSON as a string, so parse
        # that string once more before reading its fields.
        if isinstance(data, str):
            data = json.loads(data)
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(
            status_code=502, detail="AI agent API returned invalid JSON"
        ) from exc

    status = data.get("status")
    payload = data.get("payload", {})

    if status == "Valid":
        try:
            job_id = submit_yaml_job(payload.get("job"), payload.get("cmd"))
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except (ApiException, RuntimeError) as exc:
            raise HTTPException(
                status_code=503, detail=f"Unable to submit Kubernetes Job: {exc}"
            ) from exc
        return {
            "status": "success",
            "message": "Job submitted successfully",
            "job_id": job_id,
            "details": payload,
        }

    if status == "Invalid":
        raise HTTPException(
            status_code=400,
            detail=payload.get("message", "Unknown validation error."),
        )

    raise HTTPException(status_code=502, detail="AI agent API returned an unknown status")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# Routers must be included after their routes have been declared.
app.include_router(router)
