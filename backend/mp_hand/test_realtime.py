import cv2
import mediapipe as mp
import numpy as np
import json
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

app = FastAPI()

# 1. CORS 설정: 리액트(5173포트)에서 서버에 접속할 수 있도록 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. AI 모델 및 수어 단어 리스트 로드
# 수집하신 WORD1501~1520 리스트입니다.
actions = ['WORD1501', 'WORD1502', 'WORD1503', 'WORD1504', 'WORD1505', 
           'WORD1506', 'WORD1507', 'WORD1508', 'WORD1509', 'WORD1510', 
           'WORD1511', 'WORD1512', 'WORD1513', 'WORD1514', 'WORD1520']
model = load_model('suyeo_model.h5') # 모델 파일이 같은 폴더에 있어야 함

# 3. MediaPipe 초기화 (양손 인식 설정)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    프런트엔드와 연결되는 웹소켓 엔드포인트
    """
    await websocket.accept()
    cap = cv2.VideoCapture(0) # 서버(학교컴퓨터)의 웹캠 실행
    seq = [] # 62프레임을 저장할 리스트

    try:
        while cap.isOpened():
            ret, img = cap.read()
            if not ret: break
            
            img = cv2.flip(img, 1) # 좌우 반전
            # MediaPipe 처리 (BGR을 RGB로 변환)
            res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            row = []
            if res.multi_hand_landmarks:
                for i in range(2): # 양손 처리
                    if i < len(res.multi_hand_landmarks):
                        h = res.multi_hand_landmarks[i]
                        # 0번(손목) 점을 기준으로 상대 좌표 계산 (정규화)
                        base_x, base_y, base_z = h.landmark[0].x, h.landmark[0].y, h.landmark[0].z
                        for lm in h.landmark:
                            row.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                    else:
                        row.extend([0.0] * 63) # 한 손만 있으면 나머지는 0 처리
            else:
                row.extend([0.0] * 126) # 손이 아예 없으면 126개 모두 0 처리

            seq.append(row)
            if len(seq) > 62:
                seq.pop(0) # 항상 최신 62프레임 유지

            # 초기화 데이터
            result_data = {
                "word": "인식 대기 중...",
                "confidence": 0,
                "is_detected": bool(res.multi_hand_landmarks)
            }

            # 4. 데이터가 62프레임이 쌓였을 때만 AI 예측 실행
            if len(seq) == 62:
                input_data = np.expand_dims(np.array(seq, dtype=np.float32), axis=0)
                y_pred = model.predict(input_data, verbose=0).squeeze()
                i_pred = int(np.argmax(y_pred))
                
                # 신뢰도가 85% 이상일 때만 단어 확정
                if y_pred[i_pred] > 0.85:
                    result_data["word"] = actions[i_pred]
                    result_data["confidence"] = float(y_pred[i_pred])

            # 5. 프런트엔드로 결과 전송 (JSON 형태)
            await websocket.send_text(json.dumps(result_data))
            
            # CPU 점유율 조절을 위한 아주 짧은 휴식
            await asyncio.sleep(0.01)

    except Exception as e:
        print(f"연결 오류: {e}")
    finally:
        cap.release()
        print("카메라 및 소켓 종료")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)