# RoleplayAgent 워크플로우 - Mermaid 다이어그램

아래 Mermaid 코드를 [Mermaid Live Editor](https://mermaid.live/)에 복사하여 시각적 다이어그램을 확인하세요!

## 메인 실행 흐름

```mermaid
flowchart TD
    A[시작: run 메서드] --> B[1. 맥락 추출<br/>_extract_context]
    B --> C[2. 슬롯 수집<br/>_collect_slots]
    C --> D{슬롯이<br/>부족한가?}
    
    D -->|YES| E[3a. 사전 확인 질문 생성]
    E --> F[안전 스캔<br/>_safety_scan]
    F --> G{하드 임계어<br/>감지?}
    
    G -->|YES| H[mindfulness_agent<br/>위기 개입]
    G -->|NO| I[counselor_agent<br/>답변 대기]
    
    D -->|NO| J[3b. 유형 추정<br/>_infer_type]
    J --> K[4. 시나리오 생성<br/>_build_scenario]
    K --> L[5. 안전 스캔<br/>_safety_scan]
    L --> M{하드 임계어 또는<br/>위험어 ≥3개?}
    
    M -->|YES| N[mindfulness_agent<br/>위기 개입]
    M -->|NO| O[6. 상태 갱신<br/>- roleplay_count<br/>- roleplay_logs<br/>- interventions<br/>- messages]
    O --> P[counselor_agent<br/>상담사 복귀]
    
    style A fill:#e1f5fe
    style H fill:#ffebee
    style N fill:#ffebee
    style P fill:#e8f5e8
    style I fill:#fff3e0
```

## 유형 분류 로직

```mermaid
flowchart TD
    A[토픽 텍스트 입력] --> B{"과거/예전/<br/>기억/떠올라"}
    B -->|YES| C[A형: 과거 상황 재현<br/>기법: 빈 의자, 미러링, 장면 재현]
    
    B -->|NO| D{"내일/면접/<br/>발표/앞으로"}
    D -->|YES| E[B형: 미래 상황 연습<br/>기법: 미래 투사, 리허설, 대응 문장 훈련]
    
    D -->|NO| F{"관계/갈등/<br/>상사/가족"}
    F -->|YES| G[C형: 관계 역할 바꾸기<br/>기법: 역할 교대, 관계 재구성]
    
    F -->|NO| H{"이상적/이상향/<br/>원하는자아"}
    H -->|YES| I[D형: 이상적 자아 연습<br/>기법: 이상적 자아 연기, 자기대화 재구성]
    
    H -->|NO| J[A형: 기본값<br/>과거 상황 재현]
    
    style C fill:#e3f2fd
    style E fill:#f3e5f5
    style G fill:#e8f5e8
    style I fill:#fff8e1
    style J fill:#fce4ec
```

## 안전 스캔 프로세스

```mermaid
flowchart TD
    A[텍스트 입력] --> B[소문자 변환]
    B --> C[위험어 카운트<br/>_SAFETY_KEYWORDS]
    C --> D[하드 임계어 검사<br/>_HARD_RISK]
    D --> E[결과 반환<br/>카운트, 하드여부]
    
    F[위험어 목록<br/>극단, 끝내, 해치, 포기<br/>자살, 자해, 죽고, 죽을<br/>유서, 몸을던, 고통끝내] --> C
    
    G[하드 임계어<br/>자살, 자해, 죽고<br/>죽을, 유서] --> D
    
    style A fill:#e1f5fe
    style E fill:#e8f5e8
    style F fill:#ffebee
    style G fill:#ffcdd2
```

## 시나리오 생성 구조

```mermaid
flowchart TD
    A[입력 파라미터<br/>rtype, persona<br/>slots, ctx_text] --> B[유형별 기법 선택<br/>_TECHNIQUE_BY_TYPE]
    B --> C[GPT 프롬프트 구성<br/>- 페르소나<br/>- 유형/기법<br/>- 상담 맥락<br/>- 슬롯 정보]
    C --> D[GPT-4o 호출<br/>시나리오 생성]
    D --> E[3단계 템플릿<br/>1) 안전수칙<br/>2) 준비→연기<br/>3) 정리]
    
    F[페르소나 프리셋<br/>공감적 상담자<br/>단호한 코치<br/>따뜻한 친구<br/>과거의 나<br/>이상적 자아] --> C
    
    style A fill:#e1f5fe
    style E fill:#e8f5e8
    style F fill:#f3e5f5
```

## 전체 시스템 아키텍처

```mermaid
graph TB
    subgraph "RoleplayAgent 클래스"
        A[run 메서드] --> B[맥락 추출]
        B --> C[슬롯 수집]
        C --> D[유형 추정]
        D --> E[시나리오 생성]
        E --> F[안전 스캔]
    end
    
    subgraph "보조 메서드들"
        G[_extract_context]
        H[_collect_slots]
        I[_infer_type]
        J[_build_scenario]
        K[_safety_scan]
    end
    
    subgraph "외부 에이전트들"
        L[counselor_agent]
        M[mindfulness_agent]
    end
    
    subgraph "상태 관리"
        N[roleplay_count]
        O[roleplay_logs]
        P[interventions]
        Q[messages]
    end
    
    A --> G
    A --> H
    A --> I
    A --> J
    A --> K
    
    F -->|정상 완료| L
    F -->|위기 감지| M
    
    A --> N
    A --> O
    A --> P
    A --> Q
    
    style A fill:#e1f5fe
    style L fill:#e8f5e8
    style M fill:#ffebee
```

## 사용 방법

1. [Mermaid Live Editor](https://mermaid.live/)에 접속
2. 위의 Mermaid 코드 중 하나를 복사
3. 에디터에 붙여넣기
4. 시각적 다이어그램 확인!

각 다이어그램은 RoleplayAgent의 다른 측면을 보여줍니다:
- **메인 실행 흐름**: 전체적인 실행 순서와 분기점
- **유형 분류 로직**: A/B/C/D 유형 결정 과정
- **안전 스캔 프로세스**: 위험어 감지 메커니즘
- **시나리오 생성 구조**: GPT를 통한 시나리오 생성 과정
- **전체 시스템 아키텍처**: 클래스 간 관계와 데이터 흐름
