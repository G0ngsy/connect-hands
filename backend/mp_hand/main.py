import os
import sys

# [1] 필수: Protobuf 충돌 방지 (프로그램 강제 종료 막기)
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import cv2
import mediapipe as mp
import numpy as np
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# Gemini 라이브러리 로드 시 에러 방지
try:
    import google.generativeai as genai
except ImportError:
    genai = None

app = FastAPI()

# [2] CORS 설정: 프론트엔드 연결 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# [3] 설정 및 모델 로드
actions = ['안녕하세요', 'ILOVEU', '반갑다', 'HELLO', '감사합니다', '좋다', 'BAD', '만나다', 'IDLE']
#.env 파일의 내용을 환경 변수로 로드
load_dotenv()
#os.getenv를 통해 키를 안전하게 가져옴
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = None
gemini_model = None

@app.on_event("startup")
async def load_resources():
    global model, gemini_model
    try:
        # LSTM 모델 로드
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "suyeo_model.h5")
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            print("✅ 1단계: 수어 단어 인식 모델 로드 완료")
        
        # Gemini AI 설정
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            print("✅ 2단계: 문장 변환용 Gemini AI 연결 완료")
    except Exception as e:
        print(f"❌ 초기화 중 에러 발생: {e}")

# [4] MediaPipe 도구 세팅
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# [5] 메인 로직: 단어 인식 후 문장 완성
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("🚀 실시간 통신 시작")
    
    cap = cv2.VideoCapture(0)
    seq = []              # 62프레임 좌표 저장
    sentence_list = []    # 인식된 '단어'들이 차곡차곡 쌓이는 곳
    last_word = ""        # 똑같은 단어 연속 인식 방지
    translated_sentence = ""

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands_detector.process(img_rgb)

            # 좌표 추출 (생략 없이 처리)
            row = []
            if res.multi_hand_landmarks:
                for i in range(2):
                    if i < len(res.multi_hand_landmarks):
                        h = res.multi_hand_landmarks[i]
                        mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)
                        base_x, base_y = h.landmark[0].x, h.landmark[0].y
                        for lm in h.landmark:
                            row.extend([lm.x - base_x, lm.y - base_y, lm.z])
                    else:
                        row.extend([0.0] * 63)
            else:
                row.extend([0.0] * 126)

            seq.append(row)
            if len(seq) > 62: seq.pop(0)

            # 현재 화면에 보여줄 단어와 확정된 문장
            curr_word = ""
            confidence = 0.0

            # --- 로직의 핵심: 단어 인식 ---
            if len(seq) == 62 and model is not None:
                input_data = np.expand_dims(np.array(seq, dtype=np.float32), axis=0)
                y_pred = model.predict(input_data, verbose=0).squeeze()
                i_pred = int(np.argmax(y_pred))
                
                if y_pred[i_pred] > 0.90: # 90% 이상 확실할 때만 단어로 인정
                    detected_word = actions[i_pred]
                    confidence = float(y_pred[i_pred])
                    
                    if detected_word != 'IDLE':
                        curr_word = detected_word
                        # 새로운 단어가 감지되면 리스트에 추가 (단어 하나하나 인식)
                        if detected_word != last_word:
                            sentence_list.append(detected_word)
                            last_word = detected_word
                            
                            # --- 로직의 핵심: 문장 변환 ---
                            if gemini_model and len(sentence_list) >= 2:
                                try:
                                    prompt = f"수어 단어들 [{', '.join(sentence_list)}]을 매끄러운 한국어 문장으로 만들어줘."
                                    resp = gemini_model.generate_content(prompt)
                                    translated_sentence = resp.text.strip()
                                except: pass

            # 데이터 전송
            _, buffer = cv2.imencode('.jpg', frame)
            img_b64 = base64.b64encode(buffer).decode('utf-8')

            await ws.send_json({
                "word": curr_word,                  # 1단계: 지금 인식된 단어
                "confidence": confidence,
                "sentence": " + ".join(sentence_list), # 2단계: 인식된 단어들의 나열
                "translated": translated_sentence,    # 3단계: Gemini가 만든 최종 문장
                "image": img_b64
            })
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("ℹ️ 연결 종료")
    finally:
        cap.release()

if __name__ == "__main__":
    import uvicorn
    # 외부 접속 허용을 위해 0.0.0.0으로 설정
    uvicorn.run(app, host="0.0.0.0", port=8080)