from pathlib import Path

import yaml
from kubernetes import client, config


JOB_FILE = Path(__file__).with_name("dummy_job.yaml")


def submit_job() -> None:
    """Submit the dummy Kubernetes Job and exit."""
    config.load_kube_config()

    with JOB_FILE.open(encoding="utf-8") as file:
        job_manifest = yaml.safe_load(file)

    namespace = job_manifest.get("metadata", {}).get("namespace", "default")
    created_job = client.BatchV1Api().create_namespaced_job(
        namespace=namespace,
        body=job_manifest,
    )

    print(
        f"Submitted job {created_job.metadata.name!r} "
        f"to namespace {namespace!r}."
    )


if __name__ == "__main__":
    submit_job()
