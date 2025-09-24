#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
롤플레잉 에이전트 웹 테스트 인터페이스
"""

from flask import Flask, render_template, request, jsonify, session
import os
from dotenv import load_dotenv
from main import build_graph, CounselorState
import uuid
from datetime import datetime

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')

# 전역 그래프 인스턴스
graph = None

def init_graph():
    """그래프 초기화"""
    global graph
    if graph is None:
        graph = build_graph()
    return graph

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
        "waiting_for_roleplay_answer": False
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
        
        # 그래프 초기화
        graph = init_graph()
        
        # 현재 상태 가져오기
        current_state = session.get('state', {})
        
        # 사용자 메시지 추가
        current_state['messages'].append({
            'role': 'user',
            'content': user_message,
            'timestamp': datetime.now().isoformat()
        })
        
        # 그래프 실행
        print(f"그래프 실행 전 상태: {type(current_state)}")
        result = graph.invoke(current_state)
        print(f"그래프 실행 후 결과 타입: {type(result)}")
        
        # 결과를 딕셔너리로 변환 (Pydantic 모델 처리)
        try:
            if hasattr(result, 'model_dump'):
                result_dict = result.model_dump()
                print("model_dump() 사용")
            elif hasattr(result, 'dict'):
                result_dict = result.dict()
                print("dict() 사용")
            else:
                # 이미 딕셔너리인 경우
                result_dict = result
                print("이미 딕셔너리")
        except Exception as convert_error:
            print(f"변환 오류: {convert_error}")
            # 강제로 딕셔너리 변환 시도
            result_dict = {
                'messages': getattr(result, 'messages', []),
                'emotion_score': getattr(result, 'emotion_score', 0),
                'trigger_roleplay': getattr(result, 'trigger_roleplay', False),
                'roleplay_topic': getattr(result, 'roleplay_topic', ''),
                'roleplay_count': getattr(result, 'roleplay_count', 0),
                'mindfulness_count': getattr(result, 'mindfulness_count', 0),
                'next_node': getattr(result, 'next_node', ''),
                'waiting_for_roleplay_answer': getattr(result, 'waiting_for_roleplay_answer', False),
                'interventions': getattr(result, 'interventions', []),
                'roleplay_logs': getattr(result, 'roleplay_logs', [])
            }
        
        # 상태 업데이트
        session['state'] = result_dict
        
        # 응답 메시지 추출
        response_messages = []
        for msg in result_dict.get('messages', []):
            if msg.get('role') in ['counselor', 'roleplay', 'mindfulness']:
                response_messages.append({
                    'role': msg.get('role'),
                    'content': msg.get('content'),
                    'timestamp': msg.get('timestamp', datetime.now().isoformat())
                })
        
        # 최신 응답 메시지
        latest_response = response_messages[-1] if response_messages else None
        
        return jsonify({
            'success': True,
            'response': latest_response,
            'all_responses': response_messages,
            'state_info': {
                'emotion_score': result_dict.get('emotion_score', 0),
                'trigger_roleplay': result_dict.get('trigger_roleplay', False),
                'roleplay_topic': result_dict.get('roleplay_topic', ''),
                'roleplay_count': result_dict.get('roleplay_count', 0),
                'mindfulness_count': result_dict.get('mindfulness_count', 0),
                'next_node': result_dict.get('next_node', ''),
                'waiting_for_roleplay_answer': result_dict.get('waiting_for_roleplay_answer', False)
            },
            'interventions': result_dict.get('interventions', []),
            'roleplay_logs': result_dict.get('roleplay_logs', [])
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
    
    print("🚀 롤플레잉 에이전트 웹 테스트 서버 시작")
    print("📱 브라우저에서 http://localhost:5000 접속")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
