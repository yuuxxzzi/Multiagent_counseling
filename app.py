from flask import Flask, render_template, request, jsonify, session
import json
import uuid
import os
from datetime import datetime
from openai import OpenAI
from main import (
    AssistantAgent, MindfulnessAgent, RoleplayAgent, MemoryAgent,
    analyze_and_update_state, emotion_branch, gpt_emotion_analysis
)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')  # 환경변수에서 시크릿 키 가져오기

# 에이전트 초기화
assistant = AssistantAgent()
mindfulness = MindfulnessAgent()
roleplay = RoleplayAgent()
memory = MemoryAgent()

# 세션별 상태 저장
user_sessions = {}

def get_or_create_session():
    """사용자 세션을 가져오거나 새로 생성"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    session_id = session['session_id']
    
    if session_id not in user_sessions:
        user_sessions[session_id] = {
            "user_id": session_id,
            "messages": [],
            "emotion_score": 0.0,
            "extreme_keywords": [],
            "trigger_roleplay": False,
            "roleplay_topic": "",
            "auto_roleplay": False,
            "session_end": False,
            "interventions": [],
            "report": {},
            "mindfulness_count": 0,
            "roleplay_count": 0,
            "roleplay_logs": None,
            "next_node": "assistant",
            "roleplay_active": False,
            "roleplay_turn": 0,
            "roleplay_role": None
        }
    
    return user_sessions[session_id]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': '메시지가 비어있습니다.'}), 400
        
        # 세션 상태 가져오기
        state = get_or_create_session()
        
        # 사용자 메시지 추가
        state["messages"].append({"role": "user", "content": user_message})
        
        # 감정 분석 및 상태 업데이트
        state, emotion_result = analyze_and_update_state(state)
        
        # 응답 생성
        responses = []
        
        # 감정 분석 결과 추가
        responses.append({
            'type': 'emotion',
            'content': emotion_result
        })
        
        # 분기 실행
        route = emotion_branch(state)
        
        if route == "mindfulness":
            state = mindfulness.run(state)
            responses.append({
                'type': 'mindfulness',
                'content': state["messages"][-1]["content"]
            })
            
            # 롤플레잉 중에 마인드풀니스가 개입된 경우
            if state.get("roleplay_active", False):
                state = roleplay.run(state)
                responses.append({
                    'type': 'roleplay',
                    'content': state["messages"][-1]["content"]
                })
                
        elif route == "roleplay":
            state = roleplay.run(state)
            responses.append({
                'type': 'roleplay',
                'content': state["messages"][-1]["content"]
            })
        
        # 일반 상담 응답 (롤플레잉 중이 아닐 때만)
        if not state.get("roleplay_active", False):
            try:
                reply = assistant.reply(user_message)
                state["messages"].append({"role": "assistant", "content": reply})
                responses.append({
                    'type': 'assistant',
                    'content': reply
                })
            except Exception as e:
                responses.append({
                    'type': 'error',
                    'content': f'상담 응답 생성 중 오류가 발생했습니다: {str(e)}'
                })
        
        # 세션 상태 저장
        user_sessions[session['session_id']] = state
        
        return jsonify({
            'success': True,
            'responses': responses,
            'emotion': emotion_result,
            'roleplay_active': state.get("roleplay_active", False)
        })
        
    except Exception as e:
        return jsonify({'error': f'서버 오류: {str(e)}'}), 500

@app.route('/end_session', methods=['POST'])
def end_session():
    """세션 종료 및 요약 보고서 생성"""
    try:
        state = get_or_create_session()
        state["session_end"] = True
        
        # 메모리 에이전트로 요약 보고서 생성
        state = memory.run(state)
        
        # 세션 정리
        session_id = session['session_id']
        report = state.get("report", {})
        
        # 세션 데이터 삭제 (선택사항)
        if session_id in user_sessions:
            del user_sessions[session_id]
        
        return jsonify({
            'success': True,
            'report': report
        })
        
    except Exception as e:
        return jsonify({'error': f'세션 종료 중 오류: {str(e)}'}), 500

@app.route('/trigger_roleplay', methods=['POST'])
def trigger_roleplay():
    """롤플레잉 수동 트리거"""
    try:
        data = request.get_json()
        topic = data.get('topic', '')
        
        state = get_or_create_session()
        state["trigger_roleplay"] = True
        state["roleplay_topic"] = topic
        
        user_sessions[session['session_id']] = state
        
        return jsonify({'success': True, 'message': '롤플레잉이 활성화되었습니다.'})
        
    except Exception as e:
        return jsonify({'error': f'롤플레잉 트리거 중 오류: {str(e)}'}), 500

@app.route('/generate_report', methods=['POST'])
def generate_report():
    """상담 보고서 생성"""
    try:
        data = request.get_json()
        session_title = data.get('sessionTitle', '상담')
        message_count = data.get('messageCount', 0)
        emotion_data = data.get('emotionData', {})
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({'error': '상담 내용이 없습니다.'}), 400
        
        # GPT API를 사용한 보고서 생성
        from openai import OpenAI
        
        # 환경변수에서 API 키 가져오기
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return jsonify({'error': 'OpenAI API 키가 설정되지 않았습니다. OPENAI_API_KEY 환경변수를 설정해주세요.'}), 500
        
        client = OpenAI(api_key=api_key)
        
        # 상담 내용을 텍스트로 변환
        conversation_text = ""
        for msg in messages:
            role = "사용자" if msg['type'] == 'user' else "상담사"
            conversation_text += f"{role}: {msg['content']}\n"
        
        # 감정 데이터 요약
        emotion_summary = ""
        if emotion_data.get('datasets'):
            for dataset in emotion_data['datasets']:
                emotion_name = dataset['label']
                emotion_values = dataset['data']
                if emotion_values:
                    max_value = max(emotion_values)
                    avg_value = sum(emotion_values) / len(emotion_values)
                    emotion_summary += f"- {emotion_name}: 최고 {max_value:.2f}, 평균 {avg_value:.2f}\n"
        
        # GPT 프롬프트
        prompt = f"""
다음은 AI 상담 세션의 내용입니다. 이를 바탕으로 전문적이고 따뜻한 상담 보고서를 작성해주세요.

**상담 정보:**
- 상담 제목: {session_title}
- 총 메시지 수: {message_count}개
- 감정 분석 결과:
{emotion_summary}

**상담 내용:**
{conversation_text}

다음 형식으로 보고서를 작성해주세요:

## 📋 상담 보고서

### 1. 상담 개요
- 상담 주제와 주요 내용 요약

### 2. 감정 상태 분석
- 감정 변화 패턴 분석
- 주요 감정과 그 원인

### 3. 상담 진행 과정
- 상담의 흐름과 주요 전환점
- 중요한 대화 내용

### 4. 종합 평가 및 제언
- 상담자의 전반적인 상태 평가
- 향후 개선 방향 제안

보고서는 전문적이면서도 이해하기 쉽게 작성해주세요.
"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 전문 상담사입니다. 상담 내용을 분석하여 전문적이고 따뜻한 보고서를 작성해주세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        report = response.choices[0].message.content.strip()
        
        return jsonify({
            'success': True,
            'report': report
        })
        
    except Exception as e:
        return jsonify({'error': f'보고서 생성 중 오류: {str(e)}'}), 500

@app.route('/generate_session_title', methods=['POST'])
def generate_session_title():
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({'error': '상담 내용이 없습니다.'}), 400
        
        # OpenAI API 키 확인
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return jsonify({'error': 'OpenAI API 키가 설정되지 않았습니다. OPENAI_API_KEY 환경변수를 설정해주세요.'}), 500
        
        client = OpenAI(api_key=api_key)
        
        # 상담 내용을 텍스트로 변환 (처음 몇 개 메시지만)
        conversation_text = ""
        for i, msg in enumerate(messages[:6]):  # 처음 6개 메시지만 사용
            role = "사용자" if msg['type'] == 'user' else "상담사"
            conversation_text += f"{role}: {msg['content']}\n"
        
        # GPT 프롬프트
        prompt = f"""
다음은 AI 상담 세션의 초기 내용입니다. 이를 바탕으로 상담 세션의 제목을 짧고 명확하게 생성해주세요.

**상담 내용:**
{conversation_text}

상담의 주요 주제나 고민을 반영한 10-15자 이내의 간결한 제목을 생성해주세요.
예시: "직장 스트레스", "인간관계 고민", "학업 부담", "가족 갈등" 등

제목만 출력하고 다른 설명은 하지 마세요.
"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 상담 세션 제목을 생성하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.7
        )
        
        title = response.choices[0].message.content.strip()
        
        return jsonify({
            'success': True,
            'title': title
        })
        
    except Exception as e:
        print(f"제목 생성 중 오류: {e}")
        return jsonify({'error': f'제목 생성 중 오류가 발생했습니다: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
