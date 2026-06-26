# AI 멀티에이전트 상담 시스템

**AI 상담사 「벼리」**와 대화하며, 감정 분석·마인드풀니스·역할극(롤플레잉)을 상황에 맞게 연결하는 웹 기반 상담 서비스입니다.

---

## 서비스 소개

많은 사람들은 고민을 혼자 끌어안거나, 말하기 어려운 상황을 미리 연습할 기회가 없습니다. 본 프로젝트는 **OpenAI 기반 멀티에이전트**가 사용자 발화를 이해하고, 감정 상태에 따라 **일반 상담 → 안정화(마인드풀니스) → 역할극**으로 자연스럽게 전환하는 상담 경험을 제공합니다.

| 구분 | 설명 |
|------|------|
| **대상** | 일상 스트레스, 대인관계·갈등, 불안·슬픔 등을 대화로 정리하고 싶은 사용자 |
| **접근 방식** | 24시간 웹 채팅, 실시간 감정 피드백, 상황 맞춤 역할극 연습 |
| **상담사 페르소나** | UI 상 AI 상담사 이름 **「벼리」** — 공감·소크라틱 질문 중심의 대화 |
| **데이터** | 회원·세션·메시지·감정·롤플 기록을 PostgreSQL에 저장 (웹 모드) |

### 사용자 여정

1. **랜딩** — 서비스 소개 및 로그인/회원가입
2. **상담 채팅** — 벼리와 실시간 대화, 감정 분석 결과 확인
3. **개입** — 고감정·위험 신호 시 마인드풀니스, 충분한 맥락 수집 후 롤플레잉 제안/진행
4. **마무리** — 세션 종료 시 요약·보고서 생성, 감정 추이 시각화(Chart.js)

---

## 주요 기능

### 실시간 AI 상담
- 사용자 메시지에 대한 공감형 응답 및 탐색 질문
- 대화 흐름·세션 제목·상담 보고서 자동 생성

### 감정 분석
- GPT 기반 감정 분류(기쁨/슬픔/분노/불안/무기력/중립) 및 강도(0~1) 산출
- 정규식 기반 **위험 신호(극단 표현)** 탐지로 안전 분기

### 마인드풀니스 개입
- 고감정·위험 신호 감지 시 호흡·집중 안내로 안정화
- 롤플레잉 진행 중에도 필요 시 개입 가능

### 롤플레잉(역할극)
- 상담 내용에서 **시나리오 슬롯**(사건, 상대, 장소, 감정, 원인, 목표) 추출
- JSON 템플릿 RAG + 유형별 시나리오(A~D)로 맞춤 역할극
- 종료 후 **RoleplaySummaryAgent**가 인사이트 정리

### 계정·세션 관리
- 이메일 회원가입/로그인, Flask 세션 기반 인증
- 메시지·감정·개입·롤플·보고서를 DB에 영속 저장

---

## 시스템 구성 (멀티에이전트)

```
사용자 입력
    │
    ▼
감정 분석 (GPT + 위험 패턴)
    │
    ├─ 위험/고감정 ──► MindfulnessAgent
    ├─ 롤플 조건 충족 ──► RoleplayAgent ──► (종료 시) RoleplaySummaryAgent
    └─ 일반 ──► AssistantAgent (벼리)
    │
    ▼
세션 종료 ──► MemoryAgent (요약 보고서)
```

| 에이전트 | 역할 |
|----------|------|
| **AssistantAgent** | 공감·소크라틱 질문 기반 일반 상담 |
| **MindfulnessAgent** | 고감정/위험 시 안정화 스크립트 |
| **RoleplayAgent** | 템플릿·슬롯 기반 역할극 진행 |
| **RoleplaySummaryAgent** | 롤플 종료 후 상담 정리·피드백 |
| **MemoryAgent** | 세션 종료 시 종합 요약·보고서 |

롤플레잉 유형(템플릿 `TRIG-A` ~ `TRIG-D` / `SELF-D`):

| 유형 | 목적 |
|------|------|
| **A** | 과거 상황 재현·다른 관점에서 바라보기 |
| **B** | 미래/어려운 상황 대응 연습 |
| **C** | 상대방 입장 이해·공감 |
| **D** | 이상적 자아·새로운 행동 패턴 연습 |

---

## 기술 스택

### Backend
| 기술 | 용도 |
|------|------|
| **Python 3** | 애플리케이션·에이전트 로직 |
| **Flask 2.3** | 웹 서버, REST API, 세션 |
| **OpenAI API** (`openai`) | 감정 분석, 상담·롤플·보고서 생성 |
| **PostgreSQL** (`psycopg2`) | 사용자·세션·메시지·감정·슬롯·개입·보고서 |
| **python-dotenv** | 환경 변수(`.env`) |
| **LangGraph / LangChain Core** | 멀티에이전트 워크플로우 확장·실험 (의존성) |
| **Pydantic, NumPy, Matplotlib** | 데이터 검증·분석·감정 그래프(선택) |

### Frontend
| 기술 | 용도 |
|------|------|
| **HTML5 / CSS3** | 랜딩·상담·인증 UI (Vanilla) |
| **JavaScript (ES6+)** | 채팅, API 연동, 상태 관리 |
| **Chart.js** | 감정 추이 차트 |
| **Font Awesome, Google Fonts** | 아이콘·타이포(Inter, Poppins) |

### AI / 데이터
| 항목 | 내용 |
|------|------|
| **기본 모델** | `gpt-4o-mini` (환경 변수 `OPENAI_MODEL`로 변경 가능) |
| **롤플 전용** | `ROLEPLAY_MODEL` (미설정 시 `OPENAI_MODEL` 사용) |
| **RAG** | `backend/role_playing_templates/*.json` 템플릿 검색·매칭 |
| **인증** | 비밀번호 SHA-256 해시 |

### 인프라·운영
| 항목 | 내용 |
|------|------|
| **저장소** | PostgreSQL (Railway 등 `DATABASE_URL` 지원) |
| **로컬 실행** | `python run_app.py` → `http://localhost:5000` |
| **CLI 모드** | `python backend/main.py` (터미널 상담·프로토타입) |

---

## 프로젝트 구조

```
Multiagent_counseling/
├── run_app.py                 # 웹 앱 실행 진입점
├── backend/
│   ├── app.py                 # Flask 라우트·API
│   ├── db.py                  # PostgreSQL 연동
│   └── Multiagent_counseling/
│       └── main.py            # 에이전트·감정 분석·롤플 핵심 로직
├── frontend/templates/        # index, counseling, login, signup
├── images/static/             # 로고·캐릭터 등 정적 리소스
├── backend/role_playing_templates/  # 롤플 JSON 템플릿
└── README.md
```

---

## 스크린샷

실행 화면 캡처를 아래 경로에 추가한 뒤, 이미지를 연결해 주세요.

| 화면 | 예시 경로 |
|------|-----------|
| 랜딩 / 홈 | `docs/landing.png` |
| 상담 채팅 | `docs/counseling.png` |
| 감정 차트·보고서 | `docs/report.png` |

```md
![상담 화면](docs/counseling.png)
```

---

## 빠른 시작

로컬에서 웹 서비스를 띄울 때만 참고하세요. 상세 API·엔드포인트는 [README_WEB.md](README_WEB.md)를 보면 됩니다.

1. 의존성: `pip install -r backend/requirements.txt` 및 Flask 설치  
2. `.env`: `OPENAI_API_KEY`, `DATABASE_URL`(또는 DB 개별 변수), 선택 `FLASK_SECRET_KEY`  
3. 실행: `python run_app.py` → 브라우저에서 `http://localhost:5000`

---

## 참고

- 본 서비스는 **전문 의료·심리 치료를 대체하지 않습니다.** 위기 상황에서는 전문 기관·상담 전화를 이용해 주세요.
- 기술 상세·에이전트별 스펙: [기술스택_문서.md](기술스택_문서.md)
- 롤플 워크플로: [roleplay_workflow.md](roleplay_workflow.md)
