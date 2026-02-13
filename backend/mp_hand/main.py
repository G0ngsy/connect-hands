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
import google.generativeai as genai

# [1] 시스템 환경 설정
# Protobuf 충돌 방지: 특정 라이브러리 간의 버전 갈등으로 인한 크래시를 방지합니다.
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

app = FastAPI()

# CORS 설정: 프론트엔드(React)에서 백엔드로 접속할 수 있도록 허용합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# [2] 모델 및 설정 로드
# 학습 시 사용했던 단어들의 순서와 동일해야 합니다.
actions = ['안녕하세요', 'ILOVEU', '반갑다', 'HELLO', '좋다', 'BAD', '만나다', 'IDLE']
load_dotenv() # .env 파일에서 GOOGLE_API_KEY를 가져옵니다.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

model = None
gemini_model = None

@app.on_event("startup")
async def load_resources():
    """서버 시작 시 모델들을 메모리에 로드하여 실시간 처리를 준비합니다."""
    global model, gemini_model
    try:
        # LSTM 수어 인식 모델 로드
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "suyeo_model.h5")
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            print("✅ 1단계: LSTM 수어 인식 모델 로드 성공")
        
        # Gemini AI 설정 (사용자 계정에서 확인된 최신 모델명 적용)
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            # 확인된 모델명: gemini-3-flash-preview
            gemini_model = genai.GenerativeModel('gemini-3-flash-preview')
            print("✅ 2단계: Gemini 3 AI 모델 연결 성공")
    except Exception as e:
        print(f"❌ 초기화 에러: {e}")

# MediaPipe 설정 (손 가이드라인 추출)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# [3] 보조 함수: Gemini 문장 변환
async def get_gemini_response(word_list):
    """단어 리스트를 자연스러운 한국어 문장으로 변환합니다."""
    global gemini_model
    if not gemini_model: return ""
    
    prompt = f"수어 단어들 [{', '.join(word_list)}]를 자연스러운 한국어 문장으로 만들어줘. 결과 문장만 말해줘."
    try:
        # API 통신은 오래 걸리므로 스레드에서 실행하여 영상 루프가 멈추지 않게 함
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Gemini 통신 실패: {e}")
        return "문장 변환 중..."

# [4] 실시간 웹소켓 엔드포인트
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept() # 클라이언트 연결 수락
    
    cap = cv2.VideoCapture(0) # 카메라 기기 오픈
    seq = []                 # 62프레임 시퀀스 데이터를 담을 리스트
    sentence_list = []       # 인식된 단어들의 나열
    last_word = ""           # 중복 인식 방지용 변수
    translated_sentence = "" # 최종 번역 문장

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1) # 좌우 반전
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands_detector.process(img_rgb)

            # 좌표 데이터 추출 (MediaPipe)
            row = []
            if res.multi_hand_landmarks:
                for i in range(2):
                    if i < len(res.multi_hand_landmarks):
                        h = res.multi_hand_landmarks[i]
                        mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)
                        base_x, base_y = h.landmark[0].x, h.landmark[0].y
                        for lm in h.landmark:
                            row.extend([lm.x - base_x, lm.y - base_y, lm.z])
                    else: row.extend([0.0] * 63)
            else: row.extend([0.0] * 126)

            seq.append(row)
            if len(seq) > 62: seq.pop(0) # 항상 최신 62프레임 유지

            curr_word = ""
            confidence = 0.0

            # 단어 예측 로직
            if len(seq) == 62 and model is not None:
                input_data = np.expand_dims(np.array(seq, dtype=np.float32), axis=0)
                y_pred = model.predict(input_data, verbose=0).squeeze()
                i_pred = int(np.argmax(y_pred))
                
                # 예측 신뢰도가 90%를 넘을 때만 인정
                if y_pred[i_pred] > 0.90:
                    detected_word = actions[i_pred]
                    confidence = float(y_pred[i_pred])
                    
                    # IDLE이 아니고 새로운 단어일 때만 문장에 추가
                    if detected_word != 'IDLE':
                        curr_word = detected_word
                        if detected_word != last_word:
                            sentence_list.append(detected_word)
                            # 단어가 2개 이상 모이면 Gemini 번역 호출
                            if len(sentence_list) >= 2:
                                translated_sentence = await get_gemini_response(sentence_list)
                    
                    last_word = detected_word # 현재 단어를 이전 단어로 기록

            # 프레임을 Base64 이미지로 변환하여 전송
            _, buffer = cv2.imencode('.jpg', frame)
            img_b64 = base64.b64encode(buffer).decode('utf-8')

            # 프론트엔드로 실시간 데이터 전송
            await ws.send_json({
                "word": curr_word,                  # 현재 단어
                "confidence": confidence,           # 신뢰도
                "sentence": " + ".join(sentence_list), # 단어 나열
                "translated": translated_sentence,    # AI 번역 결과
                "image": img_b64                     # 영상 데이터
            })
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("ℹ️ 연결 종료")
    finally:
        cap.release()

if __name__ == "__main__":
    import uvicorn
    import os # 환경 변수를 읽기 위해 추가

    # [수정] 배포 환경(Render 등)에서 지정한 포트 번호를 가져옵니다. 
    # 로컬에서 실행할 때는 기본값으로 8080을 사용합니다.
    port = int(os.environ.get("PORT", 8080))
    
    # host를 "0.0.0.0"으로 설정해야 외부(웹)에서 접속이 가능합니다.
    uvicorn.run(app, host="0.0.0.0", port=port)