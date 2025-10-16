import os
import json
import uuid
import psycopg2
import psycopg2.extras
from datetime import datetime


def _first(keys):
    for k in keys:
        v = os.getenv(k)
        if v not in (None, ""):
            return v
    return None


def get_connection():
    """환경변수 기반으로 PostgreSQL 연결을 반환한다.

    우선순위:
    - DATABASE_URL / DATABASE_URI / DB_URL
    - 또는 (HOST/PORT/NAME/USER/PASSWORD 계열)
    - sslmode 기본 require (Railway 등 Public 접속 기본값)
    """
    database_url = _first(["DATABASE_URL", "DATABASE_URI", "DB_URL"])  # e.g. postgres://...
    sslmode = _first(["PGSSLMODE", "SSL_MODE", "SSLMODE"]) or "require"
    connect_kwargs = {"sslmode": sslmode, "connect_timeout": 5}

    # Railway 내부 호스트인 경우 로컬 프록시 포트가 있으면 host/port 오버라이드
    proxy_port = os.getenv("RAILWAY_PROXY_PORT")
    if database_url and proxy_port and "postgres.railway.internal" in database_url:
        connect_kwargs.update({"host": "127.0.0.1", "port": int(proxy_port)})

    if database_url:
        return psycopg2.connect(database_url, **connect_kwargs)

    host = _first(["DB_HOST", "PGHOST", "HOST"]) or "127.0.0.1"
    port = int(_first(["DB_PORT", "PGPORT", "PORT"]) or 5432)
    dbname = _first(["DB_NAME", "PGDATABASE", "NAME"]) or "postgres"
    user = _first(["DB_USER", "PGUSER", "USER"]) or "postgres"
    password = _first(["DB_PASSWORD", "PGPASSWORD", "PASSWORD"]) or ""
    return psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password, **connect_kwargs)


def get_table_columns(table_name: str):
    """information_schema에서 컬럼 목록과 타입을 조회한다.
    반환: {column_name: data_type}
    """
    sql = (
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
        """
    )
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(sql, (table_name,))
            rows = cur.fetchall()
    return {row["column_name"]: row["data_type"] for row in rows}


def _filter_data_for_table(table: str, data: dict) -> dict:
    cols = get_table_columns(table)
    out = {}
    for k, v in (data or {}).items():
        if k in cols:
            # json/jsonb 컬럼이면 문자열이 아닌 dict/list는 직렬화하지 않고 그대로 유지
            if cols[k].startswith("json"):
                out[k] = json.dumps(v, ensure_ascii=False) if not isinstance(v, (str, bytes)) else v
            else:
                out[k] = v
    return out


def insert_row(table: str, data: dict) -> None:
    filtered = _filter_data_for_table(table, data)
    if not filtered:
        return
    cols = ", ".join(filtered.keys())
    placeholders = ", ".join([f"%({k})s" for k in filtered.keys()])
    sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, filtered)


def update_row_by_key(table: str, key_col: str, key_val, data: dict) -> int:
    filtered = _filter_data_for_table(table, data)
    if key_col in filtered:
        filtered.pop(key_col)
    if not filtered:
        return 0
    sets = ", ".join([f"{k} = %({k})s" for k in filtered.keys()])
    sql = f"UPDATE {table} SET {sets} WHERE {key_col} = %s"
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, {**filtered},)
            cur.execute(sql, (*filtered.values(), key_val))
            return cur.rowcount


def ensure_session(session_id: str, user_id: str | None) -> None:
    """세션 레코드가 없으면 생성한다. (sessions.session_id, user_id, mode, strated_at, status)"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM sessions WHERE session_id = %s LIMIT 1", (session_id,))
            exists = cur.fetchone() is not None
            if not exists:
                cur.execute(
                    """
                    INSERT INTO sessions (session_id, user_id, mode, strated_at, status)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (session_id, user_id, "web", datetime.utcnow(), "active"),
                )


def close_session(session_id: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET status = %s, ended_at = %s WHERE session_id = %s",
                ("ended", datetime.utcnow(), session_id),
            )


def save_message(session_id: str, speaker: str, text: str) -> str:
    """messages(msg_id, session_id, speaker, text, ts) 저장 후 msg_id 반환"""
    msg_id = str(uuid.uuid4())
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO messages (msg_id, session_id, speaker, text, ts)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (msg_id, session_id, speaker, text, datetime.utcnow()),
            )
    return msg_id


def save_emotion(msg_id: str, label: str, intensity: float, model_version: str | None = None) -> None:
    emo_id = str(uuid.uuid4())
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO emotions (emo_id, msg_id, label, intensity, model_version, ts)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (emo_id, msg_id, label, float(intensity or 0.0), model_version or "", datetime.utcnow()),
            )


def upsert_slots_kv(session_id: str, slots: dict, source_msg_id: str | None = None) -> None:
    """slots 테이블은 (category, value, source_msg_id)로 관리됨. 세션 내 동일 category는 최신값으로 교체."""
    if not slots:
        return
    categories = list(slots.keys())
    with get_connection() as conn:
        with conn.cursor() as cur:
            # 기존 동일 카테고리 삭제
            cur.execute(
                f"DELETE FROM slots WHERE session_id = %s AND category = ANY(%s)",
                (session_id, categories),
            )
            # 신규 삽입
            for cat, val in slots.items():
                if val is None or str(val).strip() == "":
                    continue
                cur.execute(
                    """
                    INSERT INTO slots (slot_id, session_id, category, value, source_msg_id)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (str(uuid.uuid4()), session_id, str(cat), str(val), source_msg_id),
                )


def save_intervention(session_id: str, itype: str, content: dict | str, agent: str | None = None, trigger_flag_id: str | None = None) -> None:
    intervention_id = str(uuid.uuid4())
    payload_json = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO interventions (intervention_id, session_id, trigger_flag_id, agent, type, content, ts)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (intervention_id, session_id, trigger_flag_id, agent or "system", itype, payload_json, datetime.utcnow()),
            )


def start_roleplay_run(session_id: str, scenario_id: str | None, persona: dict | None) -> str:
    run_id = str(uuid.uuid4())
    persona_json = json.dumps(persona or {}, ensure_ascii=False)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO roleplay_runs (run_id, session_id, scenario_id, persona, started_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (run_id, session_id, scenario_id or "", persona_json, datetime.utcnow()),
            )
    return run_id


def end_roleplay_run(run_id: str, outcome: str | None = None) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE roleplay_runs SET ended_at = %s, outcome = %s WHERE run_id = %s",
                (datetime.utcnow(), outcome or "finished", run_id),
            )


def save_report(session_id: str, report: dict | str) -> None:
    # reports(summary TEXT, emotion_curve JSON, key_interventions JSON, generated_at TIMESTAMPTZ)
    summary_text = report if isinstance(report, str) else json.dumps(report, ensure_ascii=False)
    emotion_curve = None
    key_interventions = None
    try:
        if isinstance(report, dict):
            # emotion curve: score_trend
            emotion_curve = json.dumps(report.get("emotion_summary", {}).get("score_trend", []), ensure_ascii=False)
            key_interventions = json.dumps(report.get("session_overview", {}), ensure_ascii=False)
    except Exception:
        emotion_curve = None
        key_interventions = None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reports (report_id, session_id, summary, emotion_curve, key_interventions, generated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (str(uuid.uuid4()), session_id, summary_text, emotion_curve, key_interventions, datetime.utcnow()),
            )


def load_session_snapshot(session_id: str) -> dict:
    """세션 복원에 필요한 최소 정보 로드: 최근 메시지, 슬롯, 완성도."""
    snapshot = {"messages": [], "slots": {}, "completeness": 0.0}
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            try:
                cur.execute(
                    """
                    SELECT m.msg_id, m.speaker, m.text, m.ts,
                           e.label, e.intensity
                    FROM messages m
                    LEFT JOIN LATERAL (
                        SELECT label, intensity
                        FROM emotions e
                        WHERE e.msg_id = m.msg_id
                        ORDER BY ts DESC
                        LIMIT 1
                    ) e ON TRUE
                    WHERE m.session_id = %s
                    ORDER BY m.ts ASC
                    """,
                    (session_id,),
                )
                rows = cur.fetchall()
                for r in rows:
                    _msg_id = r[0]
                    speaker = r[1] or "assistant"
                    text = r[2] or ""
                    emo_label = r[4]
                    emo_intensity = r[5]
                    role = "user" if speaker == "user" else "assistant"
                    msg = {"role": role, "content": text}
                    if role == "user" and (emo_label is not None or emo_intensity is not None):
                        msg["emotion"] = {"class": emo_label or "중립", "score": float(emo_intensity or 0.0)}
                    snapshot["messages"].append(msg)
            except Exception:
                pass

            try:
                # slots는 카테고리-값 다중 행 구조 → key/value로 복구
                cur.execute(
                    "SELECT category, value FROM slots WHERE session_id = %s ORDER BY slot_id ASC",
                    (session_id,),
                )
                rows = cur.fetchall()
                slots_map = {}
                for cat, val in rows:
                    if cat in ("event", "character", "place", "emotion", "why", "goal"):
                        slots_map[cat] = val or ""
                snapshot["slots"] = slots_map
                # 완성도는 별도 컬럼이 없으므로 간이 계산
                total_keys = 6
                filled = sum(1 for k in ("event","character","place","emotion","why","goal") if slots_map.get(k))
                snapshot["completeness"] = round(filled / total_keys, 2)
            except Exception:
                pass
    return snapshot


