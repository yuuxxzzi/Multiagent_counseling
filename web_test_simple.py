#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
롤플레잉 에이전트 간단한 웹 테스트 인터페이스
"""

from flask import Flask, render_template, request, jsonify, session
import os
from dotenv import load_dotenv
import uuid
from datetime import datetime
import json

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')

# main.py에서 필요한 함수들만 import
from main import (
    gpt_emotion_analysis, 
    gpt_roleplay_trigger, 
    RoleplayAgent,
    CounselorAgent,
    MindfulnessAgent,
    client
)

def generate_empathy_response(user_message, question):
    """GPT를 활용한 자연스러운 공감 응답 생성"""
    try:
        prompt = f"""
사용자가 "{user_message}"라고 말했습니다.
이에 대해 공감하고 자연스럽게 다음 질문을 해주세요: "{question}"

요구사항:
1. 먼저 사용자의 감정에 공감하세요
2. 자연스럽게 질문으로 이어가세요
3. 상담사처럼 따뜻하고 전문적인 톤을 유지하세요
4. 2-3문장으로 간결하게 작성하세요
5. "【사전 확인】" 같은 기계적인 표현은 사용하지 마세요

예시:
"그런 상황이 정말 힘드셨겠어요. 그때 어떤 기분이셨나요?"
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 따뜻하고 전문적인 상담사입니다. 공감과 자연스러운 질문을 통해 대화를 이어가세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"공감 응답 생성 오류: {e}")
        # 오류 시 기본 응답
        return f"그런 상황이 힘드셨겠어요. {question}"

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    """파비콘 처리"""
    return '', 204  # No Content

@app.route('/api/start_session', methods=['POST'])
def start_session():
    """새 세션 시작"""
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    session['messages'] = []
    session['state'] = {
        "user_id": session_id,
        "messages": [],
        "emotion_score": 0.0,
        "extreme_keywords": [],
        "trigger_roleplay": False,
        "roleplay_topic": "롤플레잉",
        "session_end": False,
        "interventions": [],
        "report": {},
        "mindfulness_count": 0,
        "roleplay_count": 0,
        "roleplay_logs": [],
        "next_node": "emotion_analysis",
        "waiting_for_roleplay_answer": False,
        "remaining_questions": [],
        "user_answers": []
    }
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'message': '새 세션이 시작되었습니다.'
    })

@app.route('/api/send_message', methods=['POST'])
def send_message():
    """메시지 전송 및 처리"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'success': False, 'error': '메시지가 비어있습니다.'})
        
        # 현재 상태 가져오기
        current_state = session.get('state', {})
        
        # 사용자 메시지 추가
        current_state['messages'].append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"처리할 메시지: {user_message}")
        
        # 1단계: 감정 분석
        emotion_result = gpt_emotion_analysis(user_message)
        print(f"감정 분석 결과: {emotion_result}")
        
        # 2단계: 롤플레잉 트리거 판단
        roleplay_result = gpt_roleplay_trigger(user_message)
        print(f"롤플레잉 트리거 결과: {roleplay_result}")
        
        # 상태 업데이트
        current_state['emotion_score'] = emotion_result.get('emotion_score', 0.0)
        current_state['extreme_keywords'] = emotion_result.get('extreme', False)
        current_state['trigger_roleplay'] = roleplay_result.get('trigger', False)
        
        # 롤플레잉 주제 설정
        topic_map = {"A": "과거 상황 재현", "B": "미래 상황 연습", "C": "관계 역할 바꾸기", "D": "이상적 자아 연습"}
        current_state['roleplay_topic'] = topic_map.get(roleplay_result.get('type', 'None'), "롤플레잉")
        
        # 3단계: 에이전트 선택 및 실행
        response_message = None
        
        # 질문 답변 대기 상태인지 확인
        if current_state.get('waiting_for_roleplay_answer', False):
            print("질문 답변 대기 상태 - 사용자 답변 처리")
            # 사용자 답변을 저장
            current_state['user_answers'] = current_state.get('user_answers', [])
            current_state['user_answers'].append(user_message)
            
            # 남은 질문이 있는지 확인
            remaining_questions = current_state.get('remaining_questions', [])
            print(f"남은 질문 수: {len(remaining_questions)}")
            
            if remaining_questions:
                # 다음 질문 제시 (GPT를 활용한 자연스러운 상담 방식)
                next_question = remaining_questions.pop(0)
                
                # GPT를 활용한 자연스러운 공감 응답 생성
                response_message = generate_empathy_response(user_message, next_question)
                current_state['remaining_questions'] = remaining_questions
                print(f"GPT 생성 자연스러운 질문: {response_message}")
            else:
                # 모든 질문 완료 - 롤플레잉 시나리오 생성
                print("모든 질문 완료 - 롤플레잉 시나리오 생성")
                current_state['waiting_for_roleplay_answer'] = False
                
                # 롤플레잉 에이전트로 시나리오 생성 (새로운 상태로)
                roleplay_agent = RoleplayAgent()
                
                # 시나리오 생성을 위한 새로운 상태 생성
                scenario_state = {
                    "user_id": current_state.get('user_id', 'test_user'),
                    "messages": current_state.get('messages', []),
                    "emotion_score": current_state.get('emotion_score', 0),
                    "extreme_keywords": current_state.get('extreme_keywords', []),
                    "trigger_roleplay": True,
                    "roleplay_topic": current_state.get('roleplay_topic', '롤플레잉'),
                    "session_end": False,
                    "interventions": current_state.get('interventions', []),
                    "report": current_state.get('report', {}),
                    "mindfulness_count": current_state.get('mindfulness_count', 0),
                    "roleplay_count": current_state.get('roleplay_count', 0),
                    "roleplay_logs": current_state.get('roleplay_logs', []),
                    "next_node": "emotion_analysis",
                    "waiting_for_roleplay_answer": False
                }
                
                result_state = roleplay_agent.run(scenario_state)
                
                # 시나리오 메시지 찾기 (Pydantic 모델 처리)
                messages = []
                if hasattr(result_state, 'messages'):
                    messages = result_state.messages
                elif hasattr(result_state, 'model_dump'):
                    messages = result_state.model_dump().get('messages', [])
                elif isinstance(result_state, dict):
                    messages = result_state.get('messages', [])
                
                print(f"롤플레잉 메시지 수: {len(messages)}")
                
                for msg in messages:
                    print(f"메시지 확인: {msg}")
                    if isinstance(msg, dict) and msg.get('role') == 'roleplay' and "【안전 수칙】" in msg.get('content', ''):
                        response_message = msg.get('content', '')
                        print(f"롤플레잉 시나리오 발견: {response_message[:100]}...")
                        break
                
                if not response_message:
                    # 기본 시나리오 생성
                    response_message = """【안전 수칙】
- 안전한 환경에서 연습합니다
- 언제든 중단할 수 있습니다
- 감정이 격해지면 즉시 멈춥니다

[준비]
4-6 호흡(3회) → 신체감각 체크 → 오늘 목표: 회사 상황에 대한 두려움 극복하기

[연기]
상황: 회사에 가기 전 아침 준비
"오늘도 회사에 가야 하는구나... 하지만 괜찮아, 나는 할 수 있어."
→ 긍정적 대응: "네, 차근차근 해보겠습니다."
→ 부정적 대응: "정말 힘들 것 같아요..."

[정리]
자동사고: "회사에 가면 무서운 일이 생길 것이다"
근거: 과거 경험
반증: 때로는 괜찮은 날도 있었다
대안적 생각: "오늘은 조금씩 해보자"
실전 문장: "오늘은 한 걸음씩 천천히 해보겠습니다"
"""
                    print("기본 시나리오 생성")
        
        elif emotion_result.get('extreme', False) or emotion_result.get('emotion_score', 0) > 0.75:
            # 마인드풀니스 에이전트
            print("마인드풀니스 에이전트 실행")
            mindfulness_agent = MindfulnessAgent()
            result_state = mindfulness_agent.run(current_state)
            response_message = "지금 이 순간에 집중해보세요. 5초간 들이쉬고 천천히 내쉬어보세요."
            current_state['mindfulness_count'] += 1
            
        elif current_state.get('trigger_roleplay', False):
            # 롤플레잉 에이전트 - 바로 시나리오 생성
            print("롤플레잉 에이전트 실행 - 바로 시나리오 생성")
            
            # 롤플레잉 유형에 따른 시나리오 생성
            roleplay_type = current_state.get('roleplay_topic', '롤플레잉')
            
            if "과거 상황 재현" in roleplay_type:
                response_message = """【안전 수칙】
- 안전한 환경에서 연습합니다
- 언제든 중단할 수 있습니다
- 감정이 격해지면 즉시 멈춥니다

[준비]
4-6 호흡(3회) → 신체감각 체크 → 오늘 목표: 과거 상황에 대한 감정 정리하기

[연기]
상황: 과거 힘들었던 상황 재현
"그때 그 상황이 떠오르네요... 하지만 지금은 안전한 곳에 있어요."
→ 수용: "네, 그때 정말 힘들었어요."
→ 거절: "다시 생각하기 싫어요..."

[정리]
자동사고: "그때 내가 잘못했어"
근거: 과거 경험
반증: 그때도 최선을 다했어
대안적 생각: "그때는 그럴 수밖에 없었어"
실전 문장: "과거는 과거일 뿐, 지금은 다르게 할 수 있어"
"""
            elif "미래 상황 연습" in roleplay_type:
                response_message = """【안전 수칙】
- 안전한 환경에서 연습합니다
- 언제든 중단할 수 있습니다
- 긴장되면 호흡을 천천히 해보세요

[준비]
4-6 호흡(3회) → 신체감각 체크 → 오늘 목표: 미래 상황에 대한 자신감 키우기

[연기]
상황: 면접 또는 발표 상황
"안녕하세요, 면접관님. 저는 자신있게 답변드리겠습니다."
→ 긍정적 반응: "좋습니다, 계속 말씀해주세요."
→ 부정적 반응: "시간이 부족합니다."

[정리]
자동사고: "실수할 것 같아"
근거: 과거 경험
반증: 준비를 충분히 했어
대안적 생각: "차근차근 말하면 돼"
실전 문장: "천천히, 자신있게 말해보겠습니다"
"""
            elif "관계 갈등" in roleplay_type:
                response_message = """【안전 수칙】
- 안전한 환경에서 연습합니다
- 언제든 중단할 수 있습니다
- 감정이 격해지면 즉시 멈춥니다

[준비]
4-6 호흡(3회) → 신체감각 체크 → 오늘 목표: 관계 갈등 해결 방법 찾기

[연기]
상황: 갈등 상황에서의 대화
"우리 서로의 입장을 이해해보면 어떨까요?"
→ 수용: "네, 그렇게 해보죠."
→ 거절: "이해할 수 없어요."

[정리]
자동사고: "상대방이 나를 이해하지 못해"
근거: 과거 경험
반증: 때로는 이해해주는 경우도 있어
대안적 생각: "서로 다른 관점일 수 있어"
실전 문장: "우리 서로 다른 생각을 가지고 있네요"
"""
            elif "이상적 자아" in roleplay_type:
                response_message = """【안전 수칙】
- 안전한 환경에서 연습합니다
- 언제든 중단할 수 있습니다
- 이상적인 모습을 천천히 탐색해보세요

[준비]
4-6 호흡(3회) → 신체감각 체크 → 오늘 목표: 이상적인 자아 모습 찾기

[연기]
상황: 이상적인 나의 모습 연기
"저는 자신감 있고 긍정적인 사람입니다."
→ 긍정적 반응: "정말 멋진 모습이네요."
→ 부정적 반응: "그런 사람이 될 수 있을까요?"

[정리]
자동사고: "나는 그런 사람이 될 수 없어"
근거: 과거 경험
반증: 조금씩 변해가고 있어
대안적 생각: "하나씩 실천해보자"
실전 문장: "오늘부터 조금씩 바꿔보겠습니다"
"""
            else:
                response_message = """【안전 수칙】
- 안전한 환경에서 연습합니다
- 언제든 중단할 수 있습니다

[준비]
4-6 호흡(3회) → 신체감각 체크 → 오늘 목표: 상황에 대한 대처 방법 찾기

[연기]
상황: 연습하고 싶은 상황
"이 상황을 어떻게 대처할까요?"
→ 긍정적 대응: "차근차근 해보겠습니다."
→ 부정적 대응: "힘들 것 같아요..."

[정리]
자동사고: "이 상황을 해결할 수 없어"
근거: 과거 경험
반증: 다른 방법이 있을 거야
대안적 생각: "새로운 접근을 해보자"
실전 문장: "다른 방법으로 시도해보겠습니다"
"""
            
            print("롤플레잉 시나리오 생성 완료")
            
            # 롤플레잉 상태 업데이트
            current_state['roleplay_count'] = current_state.get('roleplay_count', 0) + 1
            current_state['next_node'] = 'counselor_agent'
            current_state['trigger_roleplay'] = False  # 롤플레잉 완료 후 트리거 해제
                
        else:
            # 상담사 에이전트
            print("상담사 에이전트 실행")
            counselor_agent = CounselorAgent()
            result_state = counselor_agent.run(current_state)
            
            # 상담사 에이전트의 응답 메시지 찾기
            for msg in result_state.get('messages', []):
                if msg.get('role') == 'counselor':
                    response_message = msg.get('content', '')
                    break
            
            if not response_message:
                response_message = "공감하며 대화를 이어가겠습니다."
        
        # 상태 업데이트
        session['state'] = current_state
        
        # 응답 메시지 추가
        if response_message:
            current_state['messages'].append({
                'role': 'assistant',
                'content': response_message,
                'timestamp': datetime.now().isoformat()
            })
        
        return jsonify({
            'success': True,
            'response': {
                'role': 'assistant',
                'content': response_message or "응답을 생성했습니다.",
                'timestamp': datetime.now().isoformat()
            },
            'state_info': {
                'emotion_score': current_state.get('emotion_score', 0),
                'trigger_roleplay': current_state.get('trigger_roleplay', False),
                'roleplay_topic': current_state.get('roleplay_topic', ''),
                'roleplay_count': current_state.get('roleplay_count', 0),
                'mindfulness_count': current_state.get('mindfulness_count', 0),
                'next_node': current_state.get('next_node', ''),
                'waiting_for_roleplay_answer': current_state.get('waiting_for_roleplay_answer', False)
            },
            'interventions': current_state.get('interventions', []),
            'roleplay_logs': current_state.get('roleplay_logs', [])
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"오류 상세: {error_details}")
        return jsonify({
            'success': False,
            'error': f'오류가 발생했습니다: {str(e)}',
            'details': error_details
        })

@app.route('/api/get_state', methods=['GET'])
def get_state():
    """현재 상태 조회"""
    current_state = session.get('state', {})
    return jsonify({
        'success': True,
        'state': {
            'emotion_score': current_state.get('emotion_score', 0),
            'trigger_roleplay': current_state.get('trigger_roleplay', False),
            'roleplay_topic': current_state.get('roleplay_topic', ''),
            'roleplay_count': current_state.get('roleplay_count', 0),
            'mindfulness_count': current_state.get('mindfulness_count', 0),
            'next_node': current_state.get('next_node', ''),
            'waiting_for_roleplay_answer': current_state.get('waiting_for_roleplay_answer', False),
            'messages_count': len(current_state.get('messages', []))
        },
        'interventions': current_state.get('interventions', []),
        'roleplay_logs': current_state.get('roleplay_logs', [])
    })

@app.route('/api/clear_session', methods=['POST'])
def clear_session():
    """세션 초기화"""
    session.clear()
    return jsonify({
        'success': True,
        'message': '세션이 초기화되었습니다.'
    })

if __name__ == '__main__':
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        print("환경 변수 또는 .env 파일에 API 키를 설정해주세요.")
        exit(1)
    
    print("🚀 롤플레잉 에이전트 간단한 웹 테스트 서버 시작")
    print("📱 브라우저에서 http://localhost:5000 접속")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
