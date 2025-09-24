#!/usr/bin/env python3
"""
AI 멀티에이전트 상담 시스템 웹 애플리케이션 실행 스크립트
"""

import os
import sys
from app import app

if __name__ == '__main__':
    # 환경 변수 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("   .env 파일에 OPENAI_API_KEY=your_api_key_here 를 추가하세요.")
        sys.exit(1)
    
    print("🚀 AI 멀티에이전트 상담 시스템 웹 애플리케이션을 시작합니다...")
    print("📱 브라우저에서 http://localhost:5000 으로 접속하세요.")
    print("🛑 종료하려면 Ctrl+C를 누르세요.")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 애플리케이션이 종료되었습니다.")
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")
        sys.exit(1)
