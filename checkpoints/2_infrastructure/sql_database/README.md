# PostgreSQL on Kubernetes

This checkpoint demonstrates two database workflows:

1. Deploy a PostgreSQL database on Kubernetes and inspect its resources and
   data.
2. Run `sql_example.py` locally to send SQL statements to the database through
   a Kubernetes port-forward.

The example is designed for a local, single-node teaching cluster. It uses a
Kubernetes StatefulSet and persistent storage so PostgreSQL data survives Pod
restarts.

## Files

| File | Purpose |
| --- | --- |
| `database_full.yaml` | Defines the PostgreSQL storage, configuration, credentials, Service, and StatefulSet. (All-in-one) |
| `sql_example.py` | Connects to PostgreSQL, creates a table if needed, and inserts a test row. |

## Prerequisites

- A running local Kubernetes cluster, Minikube for the class.
- `kubectl` configured to use that cluster.
- Python 3 and `pip` for the Python workflow.
- Permission to create persistent volumes, persistent volume claims, ConfigMaps, Secrets, Services, and StatefulSets (default for Minikube context).
- Port `5432` available on the local machine for port-forwarding. (Or any port available)

Check the active cluster before continuing:

```bash
kubectl config current-context
kubectl cluster-info
```

## Step 1: Deploy and Inspect PostgreSQL

From this directory, create all resources in the manifest:

```bash
kubectl apply -f database_full.yaml
```

Wait for the StatefulSet to become ready:

```bash
kubectl rollout status statefulset/test-db-statefulset --timeout=120s
```

Inspect the created resources:

```bash
kubectl get pv test-db-pv
kubectl get pvc test-db-pvc
kubectl get configmap test-db-configmap
kubectl get secret test-db-secret
kubectl get service test-db-service
kubectl get statefulset test-db-statefulset
kubectl get pods test-db-statefulset-0
```

The PVC should report `Bound`, and the Pod named
`test-db-statefulset-0` should report `Running` and `1/1` ready. For more details, inspect the Pod and its recent logs:

```bash
kubectl describe pod test-db-statefulset-0
kubectl logs test-db-statefulset-0
```

Connect to PostgreSQL from inside its Pod:

```bash
kubectl exec -it test-db-statefulset-0 -- \
  psql -U postgres -d testdb
```

At the `psql` prompt, useful inspection commands include:

```sql
\conninfo
\dt
SELECT current_database(), current_user, version();
```

To create a table, insert a row, and query the stored data manually:

```sql
CREATE TABLE IF NOT EXISTS test_table_sql (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO test_table_sql (name)
VALUES ('Manual test');

SELECT id, name, created_at
FROM test_table_sql
ORDER BY id;
```

PostgreSQL should respond with `CREATE TABLE`, `INSERT 0 1`, and a result set containing the new row.

Enter `\q` to leave `psql`.

## Step 2: Execute SQL with Python

The Python script runs outside Kubernetes, so first forward local port `5432` to the PostgreSQL Service. Keep this command running in a separate terminal:

```bash
kubectl port-forward service/test-db-service 5432:5432
```

In another terminal, create and activate a virtual environment:

```bash
python3 -m venv .venv_job
source .venv_job/bin/activate
```

Install the PostgreSQL driver and run the script:

```bash
python -m pip install psycopg2-binary
python sql_example.py
```

A successful run prints:

```text
Successfully logged event to Postgres.
```

The script performs two SQL operations:

1. Creates `test_table` if it does not already exist.
2. Inserts one row with the name `Test` and the current timestamp.

Run the script multiple times to insert additional rows.

### Verify the Inserted Data

Query the table directly through the PostgreSQL container:

```bash
kubectl exec test-db-statefulset-0 -- \
  psql -U postgres -d testdb -c \
  "SELECT * FROM test_table ORDER BY id;"
```

## Connection Settings

`sql_example.py` reads its connection settings from environment variables and
uses these defaults:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DB_HOST` | `localhost` | Database hostname. Use `localhost` with port-forwarding or `test-db-service` from another Pod. |
| `DB_PORT` | `5432` | PostgreSQL port. |
| `POSTGRES_DB` | `testdb` | Database name. |
| `POSTGRES_USER` | `postgres` | Database user. |
| `POSTGRES_PASSWORD` | `postgres` | Database password. |

For example, if local port `5432` is already occupied, forward a different
port and override `DB_PORT`:

```bash
kubectl port-forward service/test-db-service 15432:5432
DB_PORT=15432 python sql_example.py
```

## Kubernetes Resources

The manifest creates:

- A 1 GiB `hostPath` PersistentVolume named `test-db-pv`.
- A `ReadWriteOnce` PersistentVolumeClaim named `test-db-pvc`.
- A ConfigMap containing the PostgreSQL port.
- A Secret containing the database name, username, and password.
- A headless Service named `test-db-service`.
- A single-replica StatefulSet running `postgres:16`.

The PersistentVolume uses the `Retain` reclaim policy. Deleting the PVC does
not erase the database directory automatically.

## Troubleshooting

- **PVC remains `Pending`:** Run `kubectl describe pvc test-db-pvc` and verify that the PV and PVC both use the `manual` storage class and request compatible capacity and access modes.
- **Pod remains `Pending` or enters `CrashLoopBackOff`:** Inspect `kubectl describe pod test-db-statefulset-0` and `kubectl logs test-db-statefulset-0`.
- **Port-forward reports that the address is in use:** Use another local port, such as `15432`, and set `DB_PORT=15432` when running the script.
- **Python reports `connection refused`:** Confirm the database Pod is ready and the `kubectl port-forward` process is still running.
- **Python cannot import `psycopg2`:** Activate the intended virtual environment and install `psycopg2-binary` in it.
- **Table does not exist:** Run `sql_example.py` successfully at least once; the script creates the table before inserting its row.
