# AI 멀티에이전트 상담 시스템

이 프로젝트는 사용자의 발화를 바탕으로 감정 분석 → 상담 응답 → 필요 시 마인드풀니스/롤플레잉 개입을 수행하는 AI 멀티에이전트 상담 시스템입니다.

## 실행 방법

### 1) 웹(Flask) 실행

1. 가상환경 생성 및 활성화
   ```bash
   python -m venv .venv
   # Windows
   .venv\\Scripts\\activate
   ```

2. 의존성 설치
   ```bash
   pip install -r backend/requirements.txt
   pip install Flask==2.3.3 httpx==0.24.1
   ```

   - 참고: `backend/requirements_web.txt`에는 `openai`가 고정 버전으로 들어있어 충돌이 날 수 있습니다. 현재 `backend/requirements.txt`를 기준으로 `openai`를 설치하는 방식을 권장합니다.

3. 환경변수 설정
   프로젝트 루트에 `.env` 파일을 만들고 최소한 아래를 설정하세요.
   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```

   Postgres를 사용하므로 DB 연결 정보도 필요합니다(둘 중 하나 방식).

   - `DATABASE_URL` 사용(권장)
     ```env
     DATABASE_URL=postgres://USER:PASSWORD@HOST:5432/DB_NAME
     # (필요 시) PGSSLMODE=require
     ```

   - 또는 개별 키 사용
     ```env
     DB_HOST=127.0.0.1
     DB_PORT=5432
     DB_NAME=postgres
     DB_USER=postgres
     DB_PASSWORD=your_password
     ```

   선택값:
   ```env
   FLASK_SECRET_KEY=change-me
   ROLEPLAY_MODEL=gpt-4o-mini
   ```

4. 실행
   ```bash
   python run_app.py
   ```

5. 브라우저 접속
   - http://localhost:5000

### 2) 콘솔(CLI) 실행

1. 의존성 설치(콘솔도 OpenAI 호출을 하므로 동일하게 필요)
   ```bash
   pip install -r backend/requirements.txt
   ```

2. 환경변수 설정
   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```

   (콘솔 모드도 내부적으로 DB를 사용하지 않는 구조이긴 하지만, 프로젝트 공용 코드에서 `.env`를 로드하므로 OPENAI_API_KEY는 필요합니다.)

3. 실행
   ```bash
   python backend/main.py
   ```

4. 사용법
   - `입력 >` 프롬프트에 한국어 문장을 입력합니다.
   - 종료는 “빈 엔터(아무 입력 없이 Enter)”입니다.

## 기대하는 Postgres 테이블

웹 모드에서는 `backend/db.py`에서 다음 테이블들을 사용합니다(스키마는 이 프로젝트 범위 밖에서 준비되어 있어야 합니다).
- `sessions`, `users`, `messages`, `emotions`, `slots`, `interventions`
- `roleplay_runs`, `reports`

## 실행 화면 캡처(사용자 제공)

아래 섹션에 실행 화면 캡처 이미지를 넣어주세요. 예:

## 웹 실행 화면

예시 마크다운(이미지 파일을 추가한 뒤 경로만 맞춰주세요):

```md
![웹 실행 화면](docs/web-run.png)
```

## 콘솔(CLI) 실행 화면

```md
![콘솔 실행 화면](docs/cli-run.png)
```

