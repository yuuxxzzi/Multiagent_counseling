from typing import Dict, List, Any
from openai import OpenAI
import os
from langgraph.graph import StateGraph
from pydantic import BaseModel
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import re
import json
from dotenv import load_dotenv

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 상태 정의
class CounselorState(BaseModel):
    user_id: str
    messages: List[Dict[str, Any]]
    emotion_score: float
    extreme_keywords: List[str]
    trigger_roleplay: bool
    roleplay_topic: str
    session_end: bool
    interventions: List[Dict[str, Any]]
    report: Dict[str, Any]
    mindfulness_count: int = 0
    roleplay_count: int = 0
    roleplay_logs: List[str] = []
    next_node: str = "emotion_analysis"

# 안전한 JSON 파서
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

############################
# 극단성 탐지 스택 (로컬 보완)
############################
_NEGATION_RE = re.compile(r"(안|않|아니|싶지\s*않)", re.UNICODE)

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

USE_LOW_PRECISION_FAREWELL = False

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


def _extract_json_block(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON found")
    return json.loads(m.group(0))

def _normalize_snippet(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def detect_extreme_phrases(text: str) -> Dict[str, Any]:
    findings: List[Dict[str, Any]] = []
    score = 0.0
    def push(m: re.Match, cat: str, base: float) -> float:
        snippet = _normalize_snippet(m.group(0))
        findings.append({"cat": cat, "span": (m.start(), m.end()), "text": snippet})
        return base
    for pat in EXTREME_PATTERNS:
        for m in pat.finditer(text):
            score += push(m, "high_precision", 1.0)
    uniq: List[Dict[str, Any]] = []
    seen = set()
    for it in findings:
        key = (it["text"], it["cat"]) 
        if key not in seen:
            seen.add(key)
            uniq.append(it)
    return {"extreme": score >= 1.0, "score": round(score, 2), "matches": uniq, "texts": [it["text"] for it in uniq]}

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


# 감정 분석 (C 버전: 단일 호출 + 편향 방지 지시문)
def gpt_emotion_analysis(text: str) -> Dict[str, Any]:
    prompt = f"""
    다음 발화에 대해 감정 분석과 위험 신호 감지를 수행하되, 아래 JSON만 반환하세요.

    중요한 지시: 감정 분류는 위험 신호 판단과 독립적으로 수행하세요.
    - 감정_class/score는 발화의 정서적 톤만 기준으로 판단하세요.
    - 위험 신호(extreme)는 별도 판단 값으로만 표기하세요.

    발화:
    {text}

    출력(JSON):
    {{
      "emotion_class": "기쁨/슬픔/분노/불안/무기력/중립",
      "emotion_score": 0.0~1.0,
      "extreme": true 또는 false
    }}
    """

    # 로컬 보완(정규식 기반 탐지)
    local = detect_extreme_phrases(text)
    cats = detect_extreme_categories(text)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 공감 기반 감정 분석 전문가입니다. 반드시 JSON만 반환하세요. 감정 분류는 위험 신호 판단과 독립적이어야 합니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    try:
        data = _extract_json_block(response.choices[0].message.content)
    except Exception:
        data = {"emotion_class": "중립", "emotion_score": 0.0, "extreme": False}

    emotion_class = data.get("emotion_class", "중립")
    try:
        emotion_score = float(data.get("emotion_score", 0.0))
    except Exception:
        emotion_score = 0.0
    emotion_score = max(0.0, min(1.0, emotion_score))

    # 최종 extreme: '자살/직접 위험' 카테고리 매칭이 있을 때만 True
    extreme_from_cats = bool(cats.get("suicide") or cats.get("direct"))
    # 모델이 true라도 고위험 패턴 매칭이 없으면 False로 둠
    extreme_flag = extreme_from_cats or (bool(data.get("extreme")) and extreme_from_cats)

    # extreme 유형(우선순위): suicide > direct > self_denigrate > other_denigrate > indirect > seed > none
    def _pick_extreme_type(c: Dict[str, List[str]]) -> str:
        for key in ("suicide", "direct", "self_denigrate", "other_denigrate", "indirect", "seed"):
            if c.get(key):
                return key
        return "none"
    extreme_type = _pick_extreme_type(cats)

    return {
        "emotion_class": emotion_class,
        "emotion_score": emotion_score,
        "extreme": extreme_flag,
        "extreme_terms": sorted(list(set([t for key, arr in cats.items() for t in (arr if key in ("suicide", "direct") else [])]))),
        "extreme_type": extreme_type,
    }


def quick_counselor_reply(text: str) -> str:
    prompt = f"사용자의 발화: \"{text}\"\n공감하며, 소크라틱 질문을 사용해 대화를 이어가세요."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 공감과 소크라틱 질문에 능한 전문 상담자입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def analyze_and_update_state(state: Dict[str, Any]) -> Dict[str, Any]:
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

    if result["extreme"] or result["emotion_score"] > 0.75:
        if result["extreme"]:
            state["interventions"].append({
                "type": "safety_flag",
                "content": "직접적 위험 신호가 감지되어 즉시 안정화 안내로 전환합니다."
            })
        else:
            state["interventions"].append({
                "type": "high_emotion",
                "content": "감정 강도가 0.75를 초과하여 안정화 안내로 전환합니다."
            })
            state["messages"].append({
                "role": "assistant",
                "content": "지금 감정 강도가 높게 감지되었어요. 잠깐 호흡에 집중해보도록 안내할게요.",
                "agent": "notice"
            })
        state["next_node"] = "mindfulness_agent"
    else:
        state["next_node"] = "counselor_agent"

    return state


def emotion_branch(state: Dict[str, Any]) -> str:
    route = state.get("next_node", "counselor_agent")
    return route if route in ("mindfulness_agent", "counselor_agent") else "counselor_agent"

def counselor_branch(state: Dict[str, Any]) -> str:
    route = state.get("next_node", "emotion_analysis")
    if route == "counselor_agent":
        return "emotion_analysis"
    if state.get("session_end"):
        return "memory_agent"
    return "emotion_analysis"


class CounselorAgent:
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        last_input = state["messages"][-1]["content"]
        prompt = f"사용자의 발화: \"{last_input}\"\n공감하며, 소크라틱 질문을 사용해 대화를 이어가세요."
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 공감과 소크라틱 질문에 능한 전문 상담자입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        gpt_reply = response.choices[0].message.content.strip()
        state["messages"].append({"role": "counselor", "content": gpt_reply})
        return state


class MindfulnessAgent:
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        script = "지금 이 순간에 집중해보세요. 5초간 들이쉬고 천천히 내쉬어보세요."
        state["mindfulness_count"] += 1
        state["interventions"].append({"type": "mindfulness", "content": script})
        state["messages"].append({"role": "mindfulness", "content": script})
        return state


class MemoryAgent:
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]
        emotion_scores = []
        emotion_tags = []
        high_emotion_moments = []
        user_keywords = []

        session_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
        dialogue_length = len(messages)

        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                text = re.sub(r"[^\w\s]", "", msg["content"])
                user_keywords.extend(text.lower().split())
                if "emotion" in msg:
                    score = msg["emotion"]["score"]
                    emotion_scores.append(score)
                    emotion_tags.append(msg["emotion"]["class"])
                    if score > 0.7:
                        high_emotion_moments.append({"turn": i, "score": score, "text": msg["content"]})

        top_keywords = [word for word, _ in Counter(user_keywords).most_common(10)]

        plt.figure()
        plt.plot(range(len(emotion_scores)), emotion_scores, marker='o')
        plt.axhline(0.7, color='orange', linestyle='--')
        plt.axhline(0.9, color='red', linestyle='--')
        plt.title("Emotion Intensity Over Time")
        plt.xlabel("Turn Index")
        plt.ylabel("Emotion Score")
        plt.tight_layout()
        plt.savefig("emotion_graph.png")
        plt.close()

        transcript = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt = f"""
        다음 상담 대화를 요약해주세요:
        1. 대화 주제와 흐름
        2. 감정 변화 패턴
        3. 마인드풀니스 또는 롤플레잉 개입이 언제 있었는지
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

        state["report"] = {
            "session_overview": {
                "datetime": session_datetime,
                "dialogue_length": dialogue_length,
                "top_keywords": top_keywords,
                "mindfulness_used": state["mindfulness_count"],
                "roleplay_used": state["roleplay_count"]
            },
            "emotion_summary": {
                "tags": list(set(emotion_tags)),
                "score_trend": emotion_scores,
                "high_emotion_moments": high_emotion_moments,
                "graph_path": "emotion_graph.png"
            },
            "counseling_summary": parsed,
            "roleplay_details": state["roleplay_logs"]
        }

        return state


def build_graph():
    builder = StateGraph(CounselorState)
    builder.add_node("emotion_analysis", analyze_and_update_state)
    builder.add_node("counselor_agent", CounselorAgent().run)
    builder.add_node("mindfulness_agent", MindfulnessAgent().run)
    builder.add_node("memory_agent", MemoryAgent().run)

    builder.set_entry_point("emotion_analysis")

    builder.add_conditional_edges("emotion_analysis", emotion_branch, {
        "mindfulness_agent": "mindfulness_agent",
        "counselor_agent": "counselor_agent"
    })

    builder.add_edge("mindfulness_agent", "counselor_agent")
    builder.add_conditional_edges("counselor_agent", counselor_branch, {
        "memory_agent": "memory_agent",
        "emotion_analysis": "emotion_analysis"
    })

    return builder.compile()


def emotion_branch(state: Dict[str, Any]) -> str:
    route = state.get("next_node", "counselor_agent")
    return route if route in ("mindfulness_agent", "counselor_agent") else "counselor_agent"

def counselor_branch(state: Dict[str, Any]) -> str:
    route = state.get("next_node", "emotion_analysis")
    if route == "counselor_agent":
        return "emotion_analysis"
    if state.get("session_end"):
        return "memory_agent"
    return "emotion_analysis"


if __name__ == "__main__":
    print("[info][C] Building graph...")
    graph = build_graph()
    print("\nC 버전: 단일 호출 + 편향 방지 프롬프트. 한국어 문장 입력. 종료: '종료' 또는 'exit'")
    while True:
        try:
            text = input("\n입력 > ").strip()
        except EOFError:
            break
        if not text:
            continue
        if text.lower() in ("exit", "quit") or text in ("종료", "끝"):
            break

        up = gpt_emotion_analysis(text)
        print("[C]", up)
        try:
            up_reply = quick_counselor_reply(text)
            print("[C_reply]", up_reply)
        except Exception as e:
            print("[warn] C reply failed:", e)


