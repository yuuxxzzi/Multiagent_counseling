# check_pg.py
import os
import sys
import psycopg2

try:
    from dotenv import load_dotenv  # 선택 사항
    load_dotenv()
except Exception:
    pass


def main():
    def first_nonempty(keys):
        for key in keys:
            val = os.getenv(key)
            if val not in (None, ""):
                return val
        return None

    def require(keys):
        val = first_nonempty(keys)
        if val is None:
            raise RuntimeError(f"환경변수 누락: {' / '.join(keys)} 중 하나를 설정하세요")
        return val

    # DATABASE_URL 우선 지원 (없으면 키별 접속으로 폴백)
    database_url = first_nonempty(["DATABASE_URL", "DATABASE_URI", "DB_URL"])
    sslmode = first_nonempty(["PGSSLMODE", "SSL_MODE", "SSLMODE"]) or "require"  # Public 접속은 보통 require

    # .env의 두 가지 네이밍(PGHOST/PGPORT/PGDATABASE/PGUSER/PGPASSWORD, HOST/PORT/NAME/USER/PASSWORD) 모두 지원
    host_keys = ["DB_HOST", "PGHOST", "HOST"]
    port_keys = ["DB_PORT", "PGPORT", "PORT"]
    name_keys = ["DB_NAME", "PGDATABASE", "NAME"]
    user_keys = ["DB_USER", "PGUSER", "USER"]
    pass_keys = ["DB_PASSWORD", "PGPASSWORD", "PASSWORD"]

    # 내부 호스트를 사용하면서 로컬 프록시 포트가 지정된 경우 자동으로 프록시 사용
    proxy_port = os.getenv("RAILWAY_PROXY_PORT")

    try:
        if database_url:
            # DATABASE_URL 기반 연결. 프록시 포트가 지정되고 내부 호스트가 URL에 포함되면 호스트/포트를 오버라이드.
            connect_kwargs = {"sslmode": sslmode, "connect_timeout": 5}
            if proxy_port and "postgres.railway.internal" in database_url:
                connect_kwargs.update({"host": "127.0.0.1", "port": int(proxy_port)})
            conn = psycopg2.connect(database_url, **connect_kwargs)
        else:
            # 키별 접속 정보 사용 (모두 .env에서 요구)
            host = require(host_keys)
            port = int(require(port_keys))
            dbname = require(name_keys)
            user = require(user_keys)
            password = require(pass_keys)
            
            conn = psycopg2.connect(
                host=host,
                port=port,
                dbname=dbname,
                user=user,
                password=password,
                sslmode=sslmode,
                connect_timeout=5,
            )

        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    ORDER BY table_name;
                    """
                )
                rows = cur.fetchall()
                table_names = [r[0] for r in rows]
                print("연결 성공")
                print(f"테이블 개수: {len(table_names)}")
                print(f"테이블 목록: {table_names if table_names else '테이블이 없습니다'}")
    except Exception as e:
        print(f"연결 실패: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()