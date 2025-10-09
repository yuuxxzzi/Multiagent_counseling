from __future__ import annotations
from typing import Dict, List, Any, Tuple
from openai import OpenAI
import os
import glob  # 파일 검색을 위한 glob 라이브러리
# matplotlib은 선택적으로 import
MATPLOTLIB_AVAILABLE = False
# if MATPLOTLIB_AVAILABLE:
#     import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import re
import json
from dotenv import load_dotenv

# ───────────────────────────────
# Env & OpenAI
# ───────────────────────────────
load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY 가 설정되어 있지 않습니다.")

# 파인튜닝된 모델 사용 (환경변수에서 가져오거나 기본값 사용)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
print(f"[Info] 사용할 모델: {OPENAI_MODEL}")

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


def detect_extreme_categories(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {"suicide": [], "self_denigrate": [], "other_denigrate": [], "direct": [],
                                 "indirect": [], "seed": []}

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
    - emotion_class: 발화의 주된 감정을 "기쁨/슬픔/분노/불안/무기력/중립" 중 하나로 분류하세요.
    - emotion_score: 감정의 강도를 0.0 (매우 약함) 에서 1.0 (매우 강함) 사이로 평가하세요.
    - extreme: 자해, 자살 등 극단적 위험 신호가 있는지 boolean 값으로 판단하세요.

    발화: {text}
    """
    cats = detect_extreme_categories(text)

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "당신은 공감 기반 감정 분석 전문가입니다. 반드시 JSON 형식으로만 응답하세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        data = parse_json_safely(response.choices[0].message.content)
    except Exception:
        data = {"emotion_class": "중립", "emotion_score": 0.0, "extreme": False}

    emotion_score = max(0.0, min(1.0, float(data.get("emotion_score", 0.0))))
    emotion_class = data.get("emotion_class", "중립")
    extreme_from_cats = bool(cats.get("suicide") or cats.get("direct"))
    extreme_flag = extreme_from_cats or data.get("extreme", False)

    return {
        "emotion_class": emotion_class, "emotion_score": emotion_score,
        "extreme": extreme_flag,
    }


# ⬇️ [수정] RAG 및 슬롯 필링: 모든 템플릿을 하나의 리스트로 통합 관리
# ───────────────────────────────
def load_all_templates_from_directory(dir_path: str) -> List[Dict]:
    """지정된 디렉토리의 모든 JSON 파일에서 템플릿을 로드하여 하나의 리스트로 통합합니다."""
    all_templates = []
    json_files = glob.glob(os.path.join(dir_path, '*.json'))

    if not json_files:
        print(f"[Warn] '{dir_path}' 디렉토리에서 템플릿 파일을 찾을 수 없습니다.")
        return []

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                templates_in_file = json.load(f)
                all_templates.extend(templates_in_file)
            print(f"[Info] '{os.path.basename(file_path)}'에서 {len(templates_in_file)}개 템플릿 로드.")
        except Exception as e:
            print(f"[Warn] '{file_path}' 파일 로드 실패: {e}")

    return all_templates


# 프로그램 시작 시 모든 템플릿 로드
ROLE_PLAYING_TEMPLATES_DIR = "../role_playing_templates"
ALL_TEMPLATES = load_all_templates_from_directory(ROLE_PLAYING_TEMPLATES_DIR)


def retrieve_best_template(slots: Dict[str, str]) -> Dict[str, Any]:
    """통합된 전체 템플릿 리스트 내에서 슬롯과 가장 일치하는 최적의 템플릿을 찾습니다."""
    if not ALL_TEMPLATES:
        return {"template_id": "default", "scene_setup": "상황: {event}에 대해 이야기해 봅시다."}

    # 슬롯 정보를 종합하여 상황 분석
    event = slots.get('event', '')
    character = slots.get('character', '')
    emotion = slots.get('emotion', '')
    goal = slots.get('goal', '')
    
    # 상황에 따른 롤플레잉 유형 우선순위 결정
    situation_keywords = f"{event} {character} {emotion} {goal}".lower()
    
    # 유형별 우선순위 점수 계산
    type_scores = {"A": 0, "B": 0, "C": 0, "D": 0}
    
    # B 유형 (미래 상황 연습) 우선순위 키워드
    future_keywords = ["연습", "준비", "대비", "미래", "앞으로", "다음에", "확인하고", "말하고", "대화하고"]
    if any(keyword in situation_keywords for keyword in future_keywords):
        type_scores["B"] += 3
    
    # A 유형 (과거 상황 재현) 우선순위 키워드  
    past_keywords = ["재현", "다시", "그때", "과거", "이미", "벌써", "상처", "트라우마"]
    if any(keyword in situation_keywords for keyword in past_keywords):
        type_scores["A"] += 3
    
    # C 유형 (관계 역할 바꾸기) 우선순위 키워드
    empathy_keywords = ["이해", "생각", "입장", "관점", "바꿔", "상대방"]
    if any(keyword in situation_keywords for keyword in empathy_keywords):
        type_scores["C"] += 3
    
    # D 유형 (이상적 자아 연습) 우선순위 키워드
    ideal_keywords = ["되고", "바라는", "이상", "자신감", "효능감", "새로운"]
    if any(keyword in situation_keywords for keyword in ideal_keywords):
        type_scores["D"] += 3
    
    # 기본적으로 B 유형(미래 상황 연습)을 선호 (상담에서 더 유용)
    type_scores["B"] += 1
    
    # 가장 높은 점수의 유형 선택
    best_type = max(type_scores, key=type_scores.get)
    
    # 선택된 유형의 템플릿들 중에서 최적의 템플릿 찾기
    best_template = None
    max_score = -1
    
    for template in ALL_TEMPLATES:
        template_id = template.get('id', '')
        
        # 유형별 템플릿 필터링
        is_target_type = False
        if best_type == "A" and template_id.startswith('TRIG-A-'):
            is_target_type = True
        elif best_type == "B" and template_id.startswith('TRIG-B-'):
            is_target_type = True
        elif best_type == "C" and template_id.startswith('TRIG-C-'):
            is_target_type = True
        elif best_type == "D" and template_id.startswith('SELF-D'):
            is_target_type = True
            
        if not is_target_type:
            continue
            
        # 템플릿 매칭 점수 계산
        template_text = ""
        if "title" in template:
            template_text += template["title"] + " "
        if "scene_setup" in template:
            template_text += template["scene_setup"] + " "
        if "preconditions" in template and "topic_tags" in template["preconditions"]:
            template_text += " ".join(template["preconditions"]["topic_tags"]) + " "
        
        score = 0
        for keyword in situation_keywords.split():
            if keyword in template_text.lower():
                score += 1
        
        if score > max_score:
            max_score = score
            best_template = template
    
    # 매칭 점수가 낮으면 (2점 미만) 시나리오 생성
    if max_score < 2:
        print(f"[Info] 상황 분석: {situation_keywords}")
        print(f"[Info] 유형 점수: {type_scores}")
        print(f"[Info] 선택된 유형: {best_type}")
        print(f"[Info] 적합한 템플릿이 없어 시나리오를 생성합니다.")
        
        # 유형에 따른 기본 시나리오 생성
        generated_template = generate_scenario_by_type(best_type, slots)
        return generated_template
    
    print(f"[Info] 상황 분석: {situation_keywords}")
    print(f"[Info] 유형 점수: {type_scores}")
    print(f"[Info] 선택된 유형: {best_type}")
    print(f"[Info] 매칭 점수: {max_score}")
    
    return best_template


def generate_scenario_by_type(roleplay_type: str, slots: Dict[str, str]) -> Dict[str, Any]:
    """유형에 따라 사용자 상황에 맞는 시나리오를 생성합니다."""
    event = slots.get('event', '상황')
    character = slots.get('character', '상대방')
    emotion = slots.get('emotion', '감정')
    goal = slots.get('goal', '목표')
    
    if roleplay_type == "A":  # 과거 상황 재현
        return {
            "id": "GENERATED-A",
            "title": f"{event} 재현",
            "roles": {
                "rp_agent_role": character,
                "user_role": "본인"
            },
            "scene_setup": f"{event} 상황을 안전하게 재현하여 다른 관점에서 바라보는 연습",
            "objectives": ["과거 상황 재현", "다른 관점에서 바라보기", "감정 정리"],
            "constraints": ["안전한 환경 유지", "감정 조절", "건설적 대화"]
        }
    elif roleplay_type == "B":  # 미래 상황 연습
        return {
            "id": "GENERATED-B", 
            "title": f"{event} 대응 연습",
            "roles": {
                "rp_agent_role": character,
                "user_role": "본인"
            },
            "scene_setup": f"{event} 상황에 대한 대응 방법을 연습하여 자신감을 기르는 시간",
            "objectives": ["대응 방법 연습", "자신감 향상", "실전 준비"],
            "constraints": ["현실적인 상황", "건설적 피드백", "안전한 연습 환경"]
        }
    elif roleplay_type == "C":  # 관계 역할 바꾸기
        return {
            "id": "GENERATED-C",
            "title": f"{character} 입장에서 생각해보기",
            "roles": {
                "rp_agent_role": character,
                "user_role": "본인"
            },
            "scene_setup": f"{character}의 입장에서 {event} 상황을 바라보며 서로의 관점을 이해하는 시간",
            "objectives": ["상대방 입장 이해", "공감 능력 향상", "관계 개선"],
            "constraints": ["객관적 관점 유지", "상호 존중", "건설적 소통"]
        }
    elif roleplay_type == "D":  # 이상적 자아 연습
        return {
            "id": "GENERATED-D",
            "title": f"이상적인 {event} 대응",
            "roles": {
                "rp_agent_role": character,
                "user_role": "본인"
            },
            "scene_setup": f"{event} 상황에서 바라는 모습으로 행동하며 새로운 대응 방식을 연습",
            "objectives": ["이상적 자아 연습", "새로운 행동 패턴", "자기효능감 증진"],
            "constraints": ["현실적 목표 설정", "점진적 변화", "긍정적 자아상"]
        }
    else:
        # 기본 시나리오
        return {
            "id": "GENERATED-DEFAULT",
            "title": f"{event} 상황 대화",
            "roles": {
                "rp_agent_role": character,
                "user_role": "본인"
            },
            "scene_setup": f"{event}에 대해 {character}와 대화하며 상황을 해결해보는 시간",
            "objectives": ["상황 해결", "의사소통 개선", "관계 회복"],
            "constraints": ["건설적 대화", "상호 존중", "문제 해결 중심"]
        }


def update_scenario_slots(state: Dict[str, Any]) -> Dict[str, Any]:
    """대화 기록을 바탕으로 시나리오 슬롯을 채우고 state를 업데이트합니다."""
    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in state["messages"][-10:]])
    current_slots_json = json.dumps(state["scenario_slots"], ensure_ascii=False)

    prompt = f"""
    당신은 사용자의 상담 대화에서 역할극 시나리오의 핵심 요소를 '추출'하는 전문가입니다.
    주어진 대화 내용에서 각 슬롯에 해당하는 가장 구체적인 정보를 그대로 가져와 채워주세요.
    절대 추상적으로 요약하지 마세요. 사용자의 표현을 최대한 활용하세요.

    [상담 대화 내용]
    {conversation_history}

    [현재까지 채워진 슬롯 정보]
    {current_slots_json}

    [추출할 슬롯]
    - event: 사용자가 겪은 핵심적인 사건의 이름. (예: "친구와의 갈등", "면접 상황")
    - character: 사건에 관련된 상대방. (예: "나를 놀리는 친구", "압박 질문을 하는 면접관")
    - place: 사건이 발생한 구체적인 장소. (예: "학교 앞 카페", "팀 회의실")
    - emotion: 사용자가 그 상황에서 느낀 가장 두드러진 감정. (예: "서운함과 분노", "극심한 불안감")
    - why: 해당 event가 발생하게 된 '구체적인 행동이나 원인'. (예: "내 실수를 다른 사람에게 말하며 놀려서", "예상치 못한 질문을 받아서")
    - goal: 사용자가 이 상황을 통해 바라는 결과. (예: "친구에게 내 감정을 솔직하게 표현하기", "침착하게 면접 질문에 답변하기")

    반드시 아래 JSON 형식만 반환하세요.
    {{
        "event": "...", "character": "...", "place": "...",
        "emotion": "...", "why": "...", "goal": "..."
    }}
    """
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "당신은 상담 내용에서 역할극 시나리오의 구체적인 요소를 '추출'하는 분석가입니다. JSON만 반환하세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, response_format={"type": "json_object"}
        )
        updated_slots = parse_json_safely(response.choices[0].message.content)

        for key, new_value in updated_slots.items():
            if new_value:
                current_value = state["scenario_slots"].get(key)
                if current_value is None or len(str(new_value)) > len(str(current_value)):
                    state["scenario_slots"][key] = new_value

    except Exception as e:
        print(f"[Warn] 슬롯 채우기 실패: {e}")

    filled_count = sum(1 for value in state["scenario_slots"].values() if value)
    total_slots = len(state["scenario_slots"])
    state["scenario_completeness"] = round(filled_count / total_slots, 2) if total_slots > 0 else 0.0
    return state


def generate_rag_prompt(slots: Dict[str, str], template: Dict[str, Any]) -> str:
    """조회된 템플릿과 채워진 슬롯을 결합하여 최종 프롬프트를 생성합니다."""
    # 템플릿에서 사용할 수 있는 키들을 확인하고 적절한 프롬프트 생성
    if "scene_setup" in template:
        base_prompt = template["scene_setup"]
    elif "title" in template:
        base_prompt = template["title"]
    else:
        base_prompt = "상황극을 시작합니다."
    
    # 슬롯 정보를 추가하여 최종 프롬프트 생성
    slot_info = []
    for key, value in slots.items():
        if value:
            slot_info.append(f"{key}: {value}")
    
    if slot_info:
        prompt = f"{base_prompt}\n\n상황 정보: {', '.join(slot_info)}"
    else:
        prompt = base_prompt
        
    return prompt


# ───────────────────────────────
# 트리거 로직
# ───────────────────────────────
ROLEPLAY_KEYWORDS = ("롤플", "롤플레", "상황극", "역할극", "대화연습", "면접 연습", "면접", "시뮬레이션", "시나리오")


def should_trigger_roleplay(user_text: str, state: Dict[str, Any]) -> Tuple[bool, str]:
    # 최소 입력 개수 조건 확인 (5개 미만이면 롤플레잉 실행 안 함)
    user_message_count = len([msg for msg in state.get("messages", []) if msg.get("role") == "user"])
    if user_message_count < 5:
        return False, f"최소 입력 개수 미달 (현재: {user_message_count}/5)"
    
    t = user_text.strip().lower()
    if any(k in t for k in ROLEPLAY_KEYWORDS):
        return True, "사용자 요청 기반 롤플레잉"
    if state.get("trigger_roleplay"):
        return True, "플래그 기반 롤플레잉"

    if state.get("roleplay_count", 0) > 0 or state.get("roleplay_active", False):
        return False, ""

    completeness = state.get("scenario_completeness", 0.0)
    if completeness >= 0.7:
        return True, f"시나리오 완성도({int(completeness * 100)}%) 기반 자동 롤플레잉"

    return False, ""


# ───────────────────────────────
# Agents
# ───────────────────────────────
class AssistantAgent:
    def reply(self, user_text: str) -> str:
        prompt = f"사용자의 발화: \"{user_text}\"\n공감하며, 소크라틱 질문을 사용해 대화를 이어가세요."
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
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
    """대화형 롤플레잉을 진행한다."""
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        user_text = state["messages"][-1]["content"]

        if not state.get("roleplay_active"):
            # --- 롤플레잉 시작 ---
            state["roleplay_active"] = True
            state["roleplay_turn"] = 0
            state["roleplay_logs"] = []

            slots = state["scenario_slots"]
            best_template = retrieve_best_template(slots)
            
            # 롤플레잉 유형 결정 및 표시
            template_id = best_template.get('id', 'N/A')
            roleplay_type = "Unknown"
            type_description = ""
            
            if template_id.startswith('TRIG-A-'):
                roleplay_type = "A"
                type_description = "과거 상황 재현 (트라우마/상처 재현, 다른 관점에서 바라보기)"
            elif template_id.startswith('TRIG-B-'):
                roleplay_type = "B" 
                type_description = "미래 상황 연습 (어려운 상황 미리 연습, 자신감 향상)"
            elif template_id.startswith('TRIG-C-'):
                roleplay_type = "C"
                type_description = "관계 역할 바꾸기 (상대방 입장에서 생각, 공감 능력 향상)"
            elif template_id.startswith('SELF-D'):
                roleplay_type = "D"
                type_description = "이상적 자아 연습 (되고 싶은 모습으로 행동, 자기효능감 증진)"
            
            print(f"[Info] 선택된 템플릿 ID: '{template_id}'")
            print(f"[Info] 롤플레잉 유형: {roleplay_type} - {type_description}")
            
            situation_prompt = generate_rag_prompt(slots, best_template)

            # 롤플레잉 상황 제시 (AI가 바로 대화하지 않고 상황만 설명)
            user_role = best_template.get('roles', {}).get('user_role', '본인')
            ai_role = best_template.get('roles', {}).get('rp_agent_role', '상대방')
            
            situation_description = f"""
🎭 롤플레잉을 시작합니다!

📋 상황: {situation_prompt}

👥 역할 분담:
- 당신: {user_role}
- AI: {ai_role}

💡 이제 상황에 맞게 자연스럽게 대화를 시작해주세요!
({user_role} 역할로 첫 마디를 해주세요)
"""
            
            reply = situation_description
            state["roleplay_role"] = best_template.get('roles', {}).get('rp_agent_role', '상대방')
            state["user_role"] = best_template.get('roles', {}).get('user_role', '본인')
            state["roleplay_situation"] = situation_prompt

        else:
            # --- 롤플레잉 진행 중 ---
            roleplay_logs = state.get("roleplay_logs", [])
            conversation_history = "\n".join([f"{log['role']}: {log['content']}" for log in roleplay_logs[-5:]])
            ai_role = state.get("roleplay_role", "상대방")
            user_role = state.get("user_role", "본인")
            situation = state.get("roleplay_situation", "")

            # AI가 상대방 역할로 자연스럽게 응답
            response_prompt = f"""
            당신은 "{ai_role}" 역할을 맡고 있습니다.
            상황: {situation}
            
            {ai_role}로서 다음을 고려하여 자연스럽게 응답해주세요:
            - {ai_role}의 성격과 말투를 일관되게 유지
            - 실제 사람처럼 자연스러운 반응
            - 상황에 맞는 적절한 감정 표현
            - 대화의 맥락을 고려한 응답

            [최근 대화 기록]
            {conversation_history}

            [{user_role}의 최근 응답]
            "{user_text}"

            [{ai_role}의 자연스러운 응답]
            """
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", 
                     "content": f"당신은 {ai_role} 역할을 맡은 전문 연기자입니다. 실제 사람처럼 자연스럽고 현실적인 대화를 이어가세요. 과도하게 연극적이거나 부자연스러운 표현은 피하고, 일상적인 대화처럼 자연스럽게 응답하세요."},
                    {"role": "user", "content": response_prompt}
                ],
                temperature=0.8
            )
            reply = response.choices[0].message.content.strip()

        # --- 공통: 상태 업데이트 및 로그 기록 ---
        state["roleplay_turn"] += 1
        state["roleplay_logs"].append({
            "turn": state["roleplay_turn"], "role": "상대방", "content": reply,
            "at": datetime.now().isoformat(timespec="seconds")
        })
        state["messages"].append({"role": "assistant", "content": reply})

        if any(k in user_text.lower() for k in ["종료", "끝", "그만", "나가기"]):
            state["roleplay_active"] = False
            state["next_node"] = "assistant"
            state["trigger_roleplay"] = False
            state["roleplay_count"] = state.get("roleplay_count", 0) + 1
            end_message = "롤플레잉이 종료되었습니다. 일반 상담으로 돌아갑니다."
            state["messages"].append({"role": "assistant", "content": end_message})
            print(f"\n[System] {end_message}")

        return state


class MemoryAgent:
    """세션 종료 시 요약 보고서 생성"""
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]
        emotion_scores = [msg["emotion"]["score"] for msg in messages if msg.get("emotion")]
        emotion_tags = [msg["emotion"]["class"] for msg in messages if msg.get("emotion")]
        
        transcript = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        prompt = f"다음 상담 대화를 분석하고 요약 보고서를 JSON 형식으로 작성해주세요.\n\n상담:\n{transcript}"
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "너는 상담 보고서 전문가야. 반드시 JSON만 반환해."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2, response_format={"type": "json_object"}
            )
            parsed_summary = parse_json_safely(response.choices[0].message.content)
        except Exception:
            parsed_summary = {}

        report = {
            "session_overview": {
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "dialogue_length": len(messages),
                "mindfulness_used": state.get("mindfulness_count", 0),
                "roleplay_used": state.get("roleplay_count", 0)
            },
            "emotion_summary": {
                "tags": list(set(emotion_tags)),
                "score_trend": emotion_scores,
            },
            "counseling_summary": parsed_summary
        }

        if state.get("roleplay_count", 0) > 0:
            report["roleplay_details"] = {
                "scenario": state.get("scenario_slots", {}),
                "logs": state.get("roleplay_logs", [])
            }

        state["report"] = report
        return state


# ───────────────────────────────
# 상태 업데이트 & 분기
# ───────────────────────────────
def analyze_and_update_state(state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    latest_msg = state["messages"][-1]
    result = gpt_emotion_analysis(latest_msg["content"])

    latest_msg["emotion"] = result
    state["emotion_score"] = result["emotion_score"]

    # 롤플레잉 중에도 슬롯을 계속 업데이트하여 더 정확한 정보 수집
    state = update_scenario_slots(state)
    print(f"[Slots] Completeness: {state['scenario_completeness'] * 100:.0f}% | {state['scenario_slots']}")

    if result["extreme"] or result["emotion_score"] > 0.85:
        state["next_node"] = "mindfulness"
        return state, result

    do_roleplay, reason = should_trigger_roleplay(latest_msg["content"], state)
    if do_roleplay:
        state["interventions"].append({"type": "roleplay_trigger", "content": reason})
        state["next_node"] = "roleplay"
        return state, result

    state["next_node"] = "assistant"
    return state, result


def emotion_branch(state: Dict[str, Any]) -> str:
    if state.get("roleplay_active", False):
        return "roleplay"
    return state.get("next_node", "assistant")


# ───────────────────────────────
# 실행부
# ───────────────────────────────
if __name__ == "__main__":
    if not ALL_TEMPLATES:
        print("템플릿이 로드되지 않아 프로그램을 종료합니다. 'role_playing_templates' 폴더와 JSON 파일들을 확인해주세요.")
        exit()

    print("한국어 문장 입력. 종료: **빈 엔터(아무 입력 없이 Enter)**\n")

    state: Dict[str, Any] = {
        "user_id": "demo-user", "messages": [], "emotion_score": 0.0,
        "trigger_roleplay": False, "session_end": False, "interventions": [], "report": {},
        "mindfulness_count": 0, "roleplay_count": 0, "roleplay_logs": [],
        "next_node": "assistant",
        "roleplay_active": False, "roleplay_turn": 0,
        "scenario_slots": {
            "event": None, "character": None, "place": None,
            "emotion": None, "why": None, "goal": None
        },
        "scenario_completeness": 0.0,
    }

    assistant = AssistantAgent()
    mindfulness = MindfulnessAgent()
    roleplay = RoleplayAgent()
    memory = MemoryAgent()

    while True:
        try:
            text = input("\n입력 > ").strip()
        except (EOFError, KeyboardInterrupt):
            text = ""

        if not text:
            state["session_end"] = True
            state = memory.run(state)
            print("\n" + "=" * 15 + " 세션 요약 보고서(JSON) " + "=" * 15)
            print(json.dumps(state["report"], ensure_ascii=False, indent=2))
            break

        state["messages"].append({"role": "user", "content": text})

        state, result = analyze_and_update_state(state)
        print(f"[Emotion] {result}")

        route = emotion_branch(state)

        if route == "roleplay":
            state = roleplay.run(state)
            user_role = state.get("user_role", "본인")
            ai_role = state.get("roleplay_role", "상대방")
            
            # 롤플레잉 시작 시에는 상황 설명만 표시
            if not state.get("roleplay_logs"):
                print("\n[Roleplay]")
                print(state["messages"][-1]["content"])
            else:
                # 롤플레잉 진행 중에는 AI 응답 표시
                print(f"\n[{ai_role}]")
                print(state["messages"][-1]["content"])
                print(f"\n[{user_role}] 역할로 응답해주세요:")
            continue

        if route == "mindfulness":
            state = mindfulness.run(state)
            print("\n[Mindfulness]")
            print(state["messages"][-1]["content"])

        try:
            reply = assistant.reply(text)
            state["messages"].append({"role": "assistant", "content": reply})
            print("\n[Assistant]", reply)
        except Exception as e:
            print("[Warn] Assistant reply failed:", e)