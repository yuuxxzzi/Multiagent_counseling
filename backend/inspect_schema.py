import json
import os
import psycopg2
from dotenv import load_dotenv


def _first(keys):
    for k in keys:
        v = os.getenv(k)
        if v:
            return v
    return None


def main() -> None:
    load_dotenv()
    url = _first(["DATABASE_URL", "DATABASE_URI", "DB_URL"])
    sslmode = _first(["PGSSLMODE", "SSL_MODE", "SSLMODE"]) or "require"
    if url:
        conn = psycopg2.connect(url, sslmode=sslmode)
    else:
        conn = psycopg2.connect(
            host=_first(["DB_HOST", "PGHOST", "HOST"]) or "127.0.0.1",
            port=int(_first(["DB_PORT", "PGPORT", "PORT"]) or 5432),
            dbname=_first(["DB_NAME", "PGDATABASE", "NAME"]) or "postgres",
            user=_first(["DB_USER", "PGUSER", "USER"]) or "postgres",
            password=_first(["DB_PASSWORD", "PGPASSWORD", "PASSWORD"]) or "",
            sslmode=sslmode,
        )

    tables = [
        'agent_events', 'cbt_steps', 'crisis_flag', 'emotions', 'feedback', 'interventions',
        'messages', 'reports', 'roleplay_runs', 'roleplay_scenarios', 'safety_checks', 'sessions', 'slots', 'users'
    ]
    out = {}
    with conn:
        with conn.cursor() as cur:
            for t in tables:
                cur.execute(
                    """
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema='public' AND table_name=%s
                    ORDER BY ordinal_position
                    """,
                    (t,),
                )
                out[t] = cur.fetchall()
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


