import cv2
import mediapipe as mp
import numpy as np
import os
import base64
import json
import asyncio
from fastapi import FastAPI, WebSocket
from tensorflow.keras.models import load_model
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware

# 1. 초기 설정
app = FastAPI()
# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 모든 곳에서의 접속 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gemini 설정 (API 키가 없다면 빈 문자열로 두세요) ---
genai.configure(api_key="YOUR_GEMINI_API_KEY")
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- 수어 모델 및 데이터 설정 ---
actions = np.array(['나', '너', '감사합니다', '만나다', '반갑다', '안녕하세요', '사랑합니다'])
model = load_model(os.path.join('models', 'holistic_model.h5'))

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# --- 상태 관리 변수 ---
sequence = []
sentence_list = []  # 인식된 단어들의 리스트
last_translated = "" # Gemini가 번역한 마지막 문장

# 2. 핵심 함수들
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    FACE_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
    face = np.array([[results.face_landmarks.landmark[i].x, results.face_landmarks.landmark[i].y, results.face_landmarks.landmark[i].z] for i in FACE_LANDMARKS]).flatten() if results.face_landmarks else np.zeros(len(FACE_LANDMARKS)*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

async def get_gemini_translation(words):
    """단어 리스트를 자연스러운 문장으로 변환"""
    if not words: return ""
    prompt = f"다음 수어 단어들을 자연스러운 한국어 문장으로 의역해줘: {' '.join(words)}. 다른 설명 없이 문장만 출력해."
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except:
        return " ".join(words) # 에러 시 단어 나열

# 3. WebSocket 엔드포인트
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global sequence, sentence_list, last_translated
    await websocket.accept()
    
    cap = cv2.VideoCapture(0)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            # AI 추론 로직
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:] # 30프레임 유지
            
            current_word = ""
            confidence = 0.0
            
            if len(sequence) == 30:
                # predict 대신 직접 호출로 속도 개선
                res = model(np.expand_dims(sequence, axis=0), training=False)[0].numpy()
                current_word = actions[np.argmax(res)]
                confidence = float(np.max(res))
                
                # 리액트의 로그에 추가할 단어 판별 (신뢰도 90% 이상)
                if confidence > 0.90:
                    if len(sentence_list) == 0 or current_word != sentence_list[-1]:
                        sentence_list.append(current_word)
                        if len(sentence_list) > 5: sentence_list.pop(0)
                        # 단어가 새로 추가될 때만 Gemini 번역 호출
                        last_translated = await get_gemini_translation(sentence_list)

            # 리액트로 보낼 이미지 인코딩 (Base64)
            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            # 리액트의 SignResult 타입에 맞춘 데이터 패키징
            data = {
                "word": current_word if confidence > 0.8 else "IDLE",
                "confidence": confidence,
                "sentence": " ".join(sentence_list),
                "translated": last_translated,
                "image": jpg_as_text 
            }
            
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(0.03) # 약 30FPS 유지

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)