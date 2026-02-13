import os
import cv2
import mediapipe as mp
import numpy as np
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# [1] 최신 Gemini SDK로 변경 (google-genai)
try:
    from google import genai
except ImportError:
    genai = None

# Protobuf 충돌 방지
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# [2] 모델 및 설정 로드
actions = ['안녕하세요', 'ILOVEU', '반갑다', 'HELLO', '좋다', 'BAD', '만나다', 'IDLE']
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

model = None
client = None

@app.on_event("startup")
async def load_resources():
    global model, client
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "suyeo_model.h5")
        if os.path.exists(MODEL_PATH):
            # 메모리 부족 방지를 위해 compile=False 권장
            model = load_model(MODEL_PATH, compile=False)
            print("✅ 1단계: LSTM 모델 로드 성공")
        
        if GOOGLE_API_KEY and genai:
            # 최신 SDK 방식 (client 사용)
            client = genai.Client(api_key=GOOGLE_API_KEY)
            print("✅ 2단계: 최신 Gemini AI 연결 성공")
    except Exception as e:
        print(f"❌ 초기화 에러: {e}")

# [3] MediaPipe 설정 (에러 방지를 위해 초기화부 감싸기)
try:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
except Exception as e:
    print(f"⚠️ MediaPipe 초기화 실패: {e}")
    hands_detector = None

async def get_gemini_response(word_list):
    global client
    if not client: return ""
    prompt = f"수어 단어들 [{', '.join(word_list)}]를 자연스러운 한국어 문장으로 만들어줘. 결과 문장만 말해줘."
    try:
        # 최신 SDK 호출 방식
        response = await asyncio.to_thread(
            client.models.generate_content, 
            model='gemini-2.0-flash', # 최신 모델명 사용 가능
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Gemini 에러: {e}")
        return "문장 다듬는 중..."

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    # 서버 환경에서는 VideoCapture(0)가 작동하지 않으므로 
    # 프론트에서 영상 프레임을 보내줘야 하지만, 일단 구조 유지를 위해 에러 방지만 처리
    print("🚀 웹소켓 연결 성공")

    seq = []
    sentence_list = []
    last_word = ""
    translated_sentence = ""

    try:
        while True:
            # 실시간 웹 배포 시에는 카메라(VideoCapture) 대신 
            # 프론트에서 보낸 이미지를 여기서 받아 처리해야 합니다.
            # 지금은 일단 구조적 에러를 잡는 데 집중합니다.
            data = await ws.receive_json()
            # (프론트에서 이미지를 보낸다고 가정하는 로직이 필요함)
            
            await asyncio.sleep(0.1) # 루프 과열 방지
    except WebSocketDisconnect:
        print("ℹ️ 연결 종료")

if __name__ == "__main__":
    import uvicorn
    # Render 환경의 포트 바인딩을 위해 수정
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)