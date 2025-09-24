from __future__ import annotations
from typing import Dict, List, Any, Tuple
from openai import OpenAI
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import re, json
from dotenv import load_dotenv

# ───────────────────────────────
# Env & OpenAI
# ───────────────────────────────
load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY 가 설정되어 있지 않습니다.")
client = OpenAI(api_key=api_key)

# ───────────────────────────────
# 유틸
# ───────────────────────────────
def parse_json_safely(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}

def _extract_json_block(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON found")
    return json.loads(m.group(0))

def _normalize_snippet(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

# ───────────────────────────────
# 극단성 탐지 (정규식)
# ───────────────────────────────
SEED_PATTERNS = [
    r"죽\s*고\s*싶[다어어요]\b",
    r"사라지\s*고\s*싶[다어어요]\b|사라졌\s*으면\s*좋겠[다어요]",
    r"(다\s*)?끝내(고|버리)?\s*싶[다어어요]\b",
    r"없어졌\s*으면\s*좋겠[다어요]",
    r"자해(하|했|할)\w*|손목\s*긋|피\s*흘리",
]
CURATED_DIRECT = [
    r"죽\s*고\s*싶[다어어요]\b",
    r"차라리\s*죽(었\s*으면|는\s*게)\s*좋겠[다어요]\b",
    r"죽어야\s*편해(질|지)\s*것\s*같[다어어요]\b",
]
CURATED_INDIRECT = [
    r"(살|사는\s*게)\s*의미(가)?\s*없[다어어요]\b",
    r"희망(이)?\s*없[다어어요]\b",
    r"가망(이)?\s*없[다어어요]\b",
    r"(내가|나[는]?)\s*없어지(는\s*게|는\s*것[이]?|면)\s*낫겠[다어어요]\b",
    r"없어졌\s*으면\s*좋겠[다어요]\b",
    r"살아(가)?\s*기\s*싫[다어어요]\b",
    r"(다\s*)?끝내(고|버리)?\s*싶[다어어요]\b",
]
CURATED_SUICIDE = [
    r"(?<![가-힣A-Za-z0-9])자살(?![가-힣A-Za-z0-9])",
    r"자살\s*(하|할|하고|했고|하려)\w*",
    r"자살\s*하\s*고\s*싶[다어어요]\b",
    r"자살\s*할래\b|자살\s*생각\w*|자살\s*충동\w*",
    r"자살\s*마렵\w*",
    r"목숨(을)?\s*끊\w*|생(을)?\s*마감\w*",
    r"(뛰어내리|투신|목\s*매|목을\s*매)\w*",
]
CURATED_SELF_DENIGRATE = [
    r"(?:나는|난)\s*(?:병신|멍청(?:이)?|쓰레기|쓸모없(?:어|다)?|가치없(?:어|다)?|형편없(?:어|다)?)\s*(?:이야|야|다|이네|이네요|입니다|임)?",
    r"(?:나는|난)\s*(?:쓸모|가치)\s*없(?:어|다)?",
    r"나(?:는)?\s*같은\s*건\s*없어져야\s*해",
]
OTHER_DENIGRATE_PATTERNS = [
    r"(?:너|당신|쟤|걔|그놈|그년|저놈|저년|저 자식|저 사람|그 사람|새끼|놈|년|애|니들|너희|너네)\s*(?:은|는|이|가|을|를)?\s*(?:병신|멍청(?:이)?|쓰레기|형편없(?:어|다)?|쓸모없(?:어|다)?|가치없(?:어|다)?)\w*",
    r"[가-힣]{2,}\s*(?:은|는|이|가)?\s*(?:병신|멍청(?:이)?|쓰레기|형편없(?:어|다)?|쓸모없(?:어|다)?|가치없(?:어|다)?)\w*",
]

def load_extra_patterns(path: str = "extreme_patterns.json") -> List[re.Pattern]:
    pats: List[re.Pattern] = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                arr = json.load(f)
            for p in arr:
                pats.append(re.compile(p))
        except Exception:
            pass
    return pats

EXTREME_PATTERNS: List[re.Pattern] = [re.compile(p) for p in SEED_PATTERNS]
EXTREME_PATTERNS += [re.compile(p) for p in (CURATED_DIRECT + CURATED_INDIRECT + CURATED_SELF_DENIGRATE + CURATED_SUICIDE)]
EXTREME_PATTERNS += load_extra_patterns()

def detect_extreme_categories(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"suicide": [], "self_denigrate": [], "other_denigrate": [], "direct": [], "indirect": [], "seed": []}
    def collect(patterns: List[str], key: str) -> None:
        for p in patterns:
            rx = re.compile(p, re.IGNORECASE)
            for m in rx.finditer(text):
                snippet = _normalize_snippet(m.group(0))
                if snippet not in out[key]:
                    out[key].append(snippet)
    collect(SEED_PATTERNS, "seed")
    collect(CURATED_DIRECT, "direct")
    collect(CURATED_INDIRECT, "indirect")
    collect(CURATED_SUICIDE, "suicide")
    collect(CURATED_SELF_DENIGRATE, "self_denigrate")
    collect(OTHER_DENIGRATE_PATTERNS, "other_denigrate")
    return {k: v for k, v in out.items() if v}

# ───────────────────────────────
# 감정 분석 (GPT + 로컬 보완)
# ───────────────────────────────
def gpt_emotion_analysis(text: str) -> Dict[str, Any]:
    prompt = f"""
    다음 발화에 대해 감정 분석과 위험 신호 감지를 수행하되, 아래 JSON만 반환하세요.

    중요한 지시: 감정 분류는 위험 신호 판단과 독립적으로 수행하세요.
    - emotion_class/score는 발화의 정서적 톤만 기준으로 판단하세요.
    - 위험 신호(extreme)는 별도 판단 값으로만 표기하세요.

    발화:
    {text}

    출력(JSON):
    {{
      "emotion_class": "기쁨/슬픔/분노/불안/무기력/중립",
      "emotion_score": 0.0,
      "extreme": false
    }}
    """
    cats = detect_extreme_categories(text)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 공감 기반 감정 분석 전문가입니다. 반드시 JSON만 반환하세요. 감정 분류는 위험 신호 판단과 독립적이어야 합니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        data = _extract_json_block(response.choices[0].message.content)
    except Exception:
        data = {"emotion_class": "중립", "emotion_score": 0.0, "extreme": False}

    try:
        emotion_score = float(data.get("emotion_score", 0.0))
    except Exception:
        emotion_score = 0.0
    emotion_score = max(0.0, min(1.0, emotion_score))
    emotion_class = data.get("emotion_class", "중립")

    extreme_from_cats = bool(cats.get("suicide") or cats.get("direct"))
    extreme_flag = extreme_from_cats or False

    def _pick_extreme_type(c: Dict[str, List[str]]) -> str:
        for key in ("suicide", "direct", "self_denigrate", "other_denigrate", "indirect", "seed"):
            if c.get(key):
                return key
        return "none"

    return {
        "emotion_class": emotion_class,
        "emotion_score": emotion_score,
        "extreme": extreme_flag,
        "extreme_terms": sorted(list(set([t for key, arr in cats.items() for t in (arr if key in ("suicide", "direct") else [])]))),
        "extreme_type": _pick_extreme_type(cats),
    }

# ───────────────────────────────
# 트리거 로직 (키워드/플래그는 항상 허용, 휴리스틱은 옵션)
# ───────────────────────────────
ROLEPLAY_KEYWORDS = (
    "롤플", "롤플레", "상황극", "역할극", "대화연습", "면접 연습", "면접", "시뮬레이션", "시나리오"
)

def should_trigger_roleplay(user_text: str, emotion: Dict[str, Any], state: Dict[str, Any]) -> Tuple[bool, str]:
    t = user_text.strip().lower()

    # 1) 사용자 키워드 요청 → 항상 허용
    if any(k in t for k in ROLEPLAY_KEYWORDS):
        return True, "사용자 요청 기반 롤플레잉"

    # 2) 외부 플래그 → 항상 허용 (한 번 실행 뒤 자동 해제 권장)
    if state.get("trigger_roleplay"):
        topic = state.get("roleplay_topic") or "불안 상황 다루기"
        return True, f"플래그 기반 롤플레잉: {topic}"

    # 3) (옵션) 휴리스틱 자동 실행
    if state.get("auto_roleplay", False):
        cls = emotion.get("emotion_class")
        score = float(emotion.get("emotion_score", 0.0))
        is_extreme = bool(emotion.get("extreme"))
        # 원래 규칙 유지: 불안/슬픔 & 0.45~0.8 & 극단 아님
        if (cls in ("불안", "슬픔")) and (0.45 <= score <= 0.8) and not is_extreme:
            return True, "감정 휴리스틱 기반 롤플레잉"

    return False, ""

# ───────────────────────────────
# Agents
# ───────────────────────────────
class AssistantAgent:
    def reply(self, user_text: str) -> str:
        prompt = f"사용자의 발화: \"{user_text}\"\n공감하며, 소크라틱 질문을 사용해 대화를 이어가세요."
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 공감과 소크라틱 질문에 능한 전문 상담자입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

class MindfulnessAgent:
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        script = "지금 이 순간에 집중해보세요. 5초간 들이쉬고 천천히 내쉬어보세요."
        state["mindfulness_count"] = state.get("mindfulness_count", 0) + 1
        state["interventions"].append({"type": "mindfulness", "content": script})
        state["messages"].append({"role": "assistant", "content": script})
        return state

class RoleplayAgent:
    """짧은 역할극 스크립트를 생성하여 사용자의 연습을 돕는다."""
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_text = state["messages"][-1]["content"]
        topic = state.get("roleplay_topic")
        if not topic:
            if "면접" in user_text:
                topic = "면접 답변 연습"
            elif "갈등" in user_text or "싸웠" in user_text or "다퉜" in user_text:
                topic = "갈등 상황 대화 연습"
            else:
                topic = "불안 완화 대화 연습"

        prompt = f"""
        다음 주제에 대해 사용자가 따라 읽을 수 있는 6~8줄 내외의 역할극 스크립트를 만들어 주세요.
        - 주제: {topic}
        - 말투: 친절하고 구체적, 문장은 짧게.
        - 형식: "상황:", "상담자:", "나:" 라벨을 사용. 마지막 줄은 "나:"로 끝내서 사용자가 말하도록 유도.

        사용자 최근 발화: "{user_text}"
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 CBT 코치이자 역할극 진행자다. 안전하고 구체적인 스크립트를 제공해라."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        script = response.choices[0].message.content.strip()

        state["roleplay_count"] = state.get("roleplay_count", 0) + 1
        if not state.get("roleplay_logs"):
            state["roleplay_logs"] = []
        state["roleplay_logs"].append({"topic": topic, "script": script, "at": datetime.now().isoformat(timespec="seconds")})
        state["messages"].append({"role": "assistant", "content": script})

        # 다음 턴 일반 대화 복귀 + (중요) 외부 트리거 1회성 해제
        state["next_node"] = "assistant"
        state["trigger_roleplay"] = False
        return state

class MemoryAgent:
    """세션 종료 시 요약 보고서 생성"""
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]
        emotion_scores: List[float] = []
        emotion_tags: List[str] = []
        high_emotion_moments: List[Dict[str, Any]] = []
        user_keywords: List[str] = []

        session_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
        dialogue_length = len(messages)

        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                text = re.sub(r"[^\w\s]", "", msg["content"])
                user_keywords.extend(text.lower().split())
                if "emotion" in msg:
                    score = float(msg["emotion"]["score"])
                    emotion_scores.append(score)
                    emotion_tags.append(msg["emotion"]["class"])
                    if score > 0.7:
                        high_emotion_moments.append({"turn": i, "score": score, "text": msg["content"]})

        top_keywords = [word for word, _ in Counter(user_keywords).most_common(10)]

        # 감정 추이 그래프 저장
        plt.figure()
        plt.plot(range(len(emotion_scores)), emotion_scores, marker='o')
        plt.axhline(0.7, linestyle='--')
        plt.axhline(0.9, linestyle='--')
        plt.title("Emotion Intensity Over Time")
        plt.xlabel("Turn Index")
        plt.ylabel("Emotion Score")
        plt.tight_layout()
        plt.savefig("emotion_graph.png")
        plt.close()

        # 대화 요약 (GPT)
        transcript = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt = f"""
        다음 상담 대화를 요약해주세요:
        1. 대화 주제와 흐름
        2. 감정 변화 패턴
        3. 마인드풀니스 또는 롤플레잉 개입 시점
        4. 상담 종료 이유
        5. 반복되는 사고나 표현

        상담:\n{transcript}
        형식:
        {{
          "topic_summary": "...",
          "emotional_flow": "...",
          "intervention_points": [...],
          "repeated_patterns": "...",
          "session_end_reason": "..."
        }}
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "너는 상담 보고서 전문가야. 반드시 JSON만 반환해."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            parsed = parse_json_safely(response.choices[0].message.content)
        except Exception:
            parsed = {}

        report = {
            "session_overview": {
                "datetime": session_datetime,
                "dialogue_length": dialogue_length,
                "top_keywords": top_keywords,
                "mindfulness_used": state.get("mindfulness_count", 0),
                "roleplay_used": state.get("roleplay_count", 0)
            },
            "emotion_summary": {
                "tags": list(set(emotion_tags)),
                "score_trend": emotion_scores,
                "high_emotion_moments": high_emotion_moments,
                "graph_path": "emotion_graph.png"
            },
            "counseling_summary": parsed
        }

        # 실제로 롤플이 있었을 때만 포함
        if state.get("roleplay_count", 0) > 0 and state.get("roleplay_logs"):
            report["roleplay_details"] = state["roleplay_logs"]

        state["report"] = report
        return state

# ───────────────────────────────
# 상태 업데이트 & 분기
# ───────────────────────────────
def analyze_and_update_state(state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    latest_msg = state["messages"][-1]
    result = gpt_emotion_analysis(latest_msg["content"])

    latest_msg["emotion"] = {
        "class": result["emotion_class"],
        "score": result["emotion_score"],
        "extreme": result["extreme"]
    }
    state["messages"][-1] = latest_msg
    state["emotion_score"] = result["emotion_score"]
    state["extreme_keywords"] = result.get("extreme_terms", [])

    # 1) 위험/고감정이면 마인드풀니스
    if result["extreme"] or result["emotion_score"] > 0.75:
        state["interventions"].append({
            "type": "safety_or_high_emotion",
            "content": "안정화 안내가 필요합니다."
        })
        state["next_node"] = "mindfulness"
        return state, result

    # 2) 롤플레잉 조건 검사
    do_roleplay, reason = should_trigger_roleplay(latest_msg["content"], result, state)
    if do_roleplay:
        if reason:
            state["interventions"].append({"type": "roleplay_trigger", "content": reason})
        state["next_node"] = "roleplay"
        return state, result

    # 3) 일반 대화
    state["next_node"] = "assistant"
    return state, result

def emotion_branch(state: Dict[str, Any]) -> str:
    route = state.get("next_node", "assistant")
    return route if route in ("assistant", "mindfulness", "roleplay") else "assistant"

# ───────────────────────────────
# 실행부
# ───────────────────────────────
if __name__ == "__main__":
    print("한국어 문장 입력. 종료: **빈 엔터(아무 입력 없이 Enter)**\n")

    state: Dict[str, Any] = {
        "user_id": "demo-user",
        "messages": [],
        "emotion_score": 0.0,
        "extreme_keywords": [],
        "trigger_roleplay": False,  # 외부에서 True 지정 시 즉시 1회 실행
        "roleplay_topic": "",       # 있으면 해당 주제로 롤플
        "auto_roleplay": False,     # 휴리스틱 자동 롤플 (기본 OFF 권장)
        "session_end": False,
        "interventions": [],
        "report": {},
        "mindfulness_count": 0,
        "roleplay_count": 0,
        "roleplay_logs": None,      # 실제 실행 전까지 None
        "next_node": "assistant",
    }

    assistant = AssistantAgent()
    mindfulness = MindfulnessAgent()
    roleplay = RoleplayAgent()
    memory = MemoryAgent()

    while True:
        try:
            text = input("\n입력 > ")
        except EOFError:
            break

        # ✅ 빈 엔터로 종료
        if text is None or text.strip() == "":
            state["session_end"] = True
            state = memory.run(state)
            print("\n=== 세션 요약 보고서(JSON) ===")
            print(json.dumps(state["report"], ensure_ascii=False, indent=2))
            print("\n그래프 파일 저장: emotion_graph.png")
            break

        text = text.strip()

        # 사용자 메시지 누적
        state["messages"].append({"role": "user", "content": text})

        # 감정 분석 (+ 즉시 출력)
        state, result = analyze_and_update_state(state)
        print(f"[C] {result}")

        # 분기 실행 + 실행 내용도 즉시 출력
        route = emotion_branch(state)
        if route == "mindfulness":
            state = mindfulness.run(state)
            print("\n[Mindfulness]")
            print(state["messages"][-1]["content"])
        elif route == "roleplay":
            state = roleplay.run(state)
            print("\n[Roleplay]")
            print(state["messages"][-1]["content"])

        # assistant 응답 생성 (+ 즉시 출력)
        try:
            reply = assistant.reply(text)
            state["messages"].append({"role": "assistant", "content": reply})
            print("\n[A]", reply)
        except Exception as e:
            print("[warn] assistant reply failed:", e)
