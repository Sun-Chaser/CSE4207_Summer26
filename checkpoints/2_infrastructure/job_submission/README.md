# Kubernetes Job Submission

This checkpoint demonstrates two ways to submit the same one-time Job to a Kubernetes cluster:

1. Submit `dummy_job.yaml` directly with `kubectl`.
2. Run `job.py`, which submits the manifest through the Kubernetes Python
   client.

The Job starts a BusyBox container, prints a success message, and exits. Each submission receives a unique name such as `dummy-job-abc12`. Completed Jobs are automatically deleted after 60 seconds.

## Files

| File | Purpose |
| --- | --- |
| `dummy_job.yaml` | Kubernetes Job manifest shared by both submission methods. |
| `job.py` | Loads the YAML manifest and submits it with the Kubernetes Python client. |
| `requirements.txt` | Python dependencies required by `job.py`. |

## Prerequisites

Before using either method, you need:

- A running Kubernetes cluster, Minikube in this class.
- A valid kubeconfig with the intended cluster selected as the current context.
- Permission to create Jobs in the `default` namespace.
- Access to pull the `busybox:1.36` container image.

Confirm the current cluster and your permissions (root user should have the following permissions by default):

```bash
kubectl config current-context
kubectl cluster-info
kubectl auth can-i create jobs --namespace default
```

The final command should print `yes`.

## Step 1: Submit Directly with `kubectl`

From this directory, create the Job and watch the status:

```bash
kubectl apply -f dummy_job_kubectl.yaml
kubectl get pods -w
```
Read the output from its Pod:

```bash
kubectl logs -l app=dummy-job --tail=-1
```

The expected message is:

```text
Dummy job completed successfully.
```

You can also inspect the Job and its Pod before the automatic cleanup occurs:

```bash
kubectl describe pod dummy-job-kubectl
kubectl get pods -l app=dummy-job
```

## Step 2: Submit with Python

Create and activate a virtual environment:

```bash
python3 -m venv .venv_job
source .venv_job/bin/activate
```

Install the dependencies and run the submission script:

```bash
python -m pip install -r requirements.txt
python job.py
```

The script:

1. Loads the active context from the local kubeconfig.
2. Reads `dummy_job.yaml` from the script's directory.
3. Uses the namespace in the manifest, which is `default`.
4. Creates the Job through Kubernetes's Batch API.

A successful submission prints a message similar to:

```text
Submitted job 'dummy-job-abc12' to namespace 'default'.
```

Copy the printed Job name to monitor it:

```bash
kubectl wait --for=condition=complete job/dummy-job-abc12 --timeout=60s
kubectl logs -l app=dummy-job --tail=-1
```

Replace `dummy-job-abc12` with the name printed by `job.py`.

## Manifest Behavior

The example Job has the following settings:

- `restartPolicy: Never` prevents the Pod from restarting its container.
- `backoffLimit: 0` prevents Kubernetes from retrying a failed Job.
- `ttlSecondsAfterFinished: 60` removes the Job about 60 seconds after it finishes.
- `generateName: dummy-job-` allows repeated submissions without name collisions.

These settings make the manifest useful for demonstrating one-shot workloads, where a task runs once, reports its result, and terminates.

## Troubleshooting

- **Connection or kubeconfig error:** Run `kubectl cluster-info` and verify that the correct context is selected.
- **`Forbidden` response:** The current Kubernetes identity does not have
  permission to create Jobs in the `default` namespace.
- **Pod remains in `Pending`:** Inspect it with `kubectl describe pod <pod-name>` to check scheduling and image-pull events.
- **Job or logs disappear:** The TTL controller deletes the completed Job and its Pod after 60 seconds. Submit it again or temporarily increase `ttlSecondsAfterFinished` while debugging. This function is disabled for file `dummy_job_kubectl.yaml`
- **Python cannot find the manifest:** Keep `dummy_job.yaml` beside `job.py`. The script resolves the YAML path relative to its own file, so it can be run from another working directory.
