# Run kubectl port-forward service/test-db-service 5432:5432 before running this

# SQL Database
import psycopg2
from psycopg2.extras import Json
import os


def get_db_connection():
    return psycopg2.connect(
        # Use localhost for local runs with `kubectl port-forward`.
        # Pods in the cluster can set DB_HOST=test-db-service.
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "testdb"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        connect_timeout=5,
    )

def log_event():
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS test_table (
                        id SERIAL PRIMARY KEY,
                        name TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cursor.execute(
                    """
                    INSERT INTO test_table (name)
                    VALUES (%s)
                    """,
                    ("Test",),
                )
                print("Successfully logged event to Postgres.")
    except psycopg2.Error as error:
        print(f"Warning: Failed to log metrics to Postgres: {error}")

def __main__():
    log_event()

if __name__ == "__main__":
    __main__()
