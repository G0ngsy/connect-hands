import cv2
import os
import mediapipe as mp
import numpy as np
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# [1] 모델 로드 설정
model = None
actions = ['WORD1501', 'WORD1502', 'WORD1503', 'WORD1504', 'WORD1505', 
           'WORD1506', 'WORD1507', 'WORD1508', 'WORD1509', 'WORD1510', 
           'WORD1511', 'WORD1512', 'WORD1513', 'WORD1514', 'WORD1518', 
           'WORD1519', 'WORD1520','LOVE','GLAD','HELLO', 'THANKS']

@app.on_event("startup")
def load_resources():
    global model
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(BASE_DIR, "suyeo_model.h5")
        model = load_model(MODEL_PATH)
        print("✅ AI 모델 로드 성공!")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")

# [2] MediaPipe 설정
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("🚀 클라이언트와 실시간 통신 시작!")
    
    # 카메라 연결 시도 (0번이 안되면 1번으로 자동 전환 시도)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("❌ 카메라를 찾을 수 없습니다.")
        await ws.send_json({"word": "Camera Error", "confidence": 0})
        await ws.close()
        return

    seq = []
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands_detector.process(img_rgb)

            row = []
            if res.multi_hand_landmarks:
                for i in range(2):
                    if i < len(res.multi_hand_landmarks):
                        h = res.multi_hand_landmarks[i]
                        mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)
                        base_x, base_y, base_z = h.landmark[0].x, h.landmark[0].y, h.landmark[0].z
                        for lm in h.landmark:
                            row.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                    else:
                        row.extend([0.0] * 63)
            else:
                row.extend([0.0] * 126)

            seq.append(row)
            if len(seq) > 62: seq.pop(0)

            word = "Waiting..."
            confidence = 0.0

            # 62프레임이 모였고 모델이 정상 로드되었을 때만 예측
            if len(seq) == 62 and model is not None:
                input_data = np.expand_dims(np.array(seq, dtype=np.float32), axis=0)
                y_pred = model.predict(input_data, verbose=0).squeeze()
                i_pred = int(np.argmax(y_pred))
                if y_pred[i_pred] > 0.85:
                    word = actions[i_pred]
                    confidence = float(y_pred[i_pred])

            # 영상 전송 준비
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')

            # 데이터 전송
            await ws.send_json({
                "word": word,
                "confidence": confidence,
                "image": img_base64,
                "is_detected": bool(res.multi_hand_landmarks)
            })
            # 프레임 속도 조절 (너무 빠르면 브라우저가 힘들어함)
            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        print("ℹ️ 클라이언트 연결 종료")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
    finally:
        cap.release()
        print("🎥 카메라 자원 해제")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)