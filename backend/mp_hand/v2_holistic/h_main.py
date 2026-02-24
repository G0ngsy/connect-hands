import cv2
import mediapipe as mp
import numpy as np
import os
import base64
import json
import asyncio
import time
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import google.generativeai as genai
from dotenv import load_dotenv

# .env 파일 로드 (API 키 보안)
load_dotenv()

# ==========================================
# 1. 초기 설정 및 라이브러리 세팅
# ==========================================
app = FastAPI()

# 브라우저(React) 접속 허용 설정 (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gemini AI API 설정 ---
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print("✅ Gemini API 연결 성공!")
else:
    print("❌ .env 파일에서 GOOGLE_API_KEY를 찾을 수 없습니다.")

# ==========================================
# 2. 모델 및 수어 데이터 로드
# ==========================================
# 실제 data 폴더의 폴더명들을 읽어와 가나다순으로 라벨링합니다.
DATA_PATH = os.path.join('data')
actions = np.array(sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]))
print(f"✅ 인식 가능한 단어 리스트 ({len(actions)}개): {actions}")

# 학습된 LSTM 모델 로드
model = load_model(os.path.join('models', 'holistic_model.h5'))

# MediaPipe Holistic 모델 초기화
mp_holistic = mp.solutions.holistic
#그림을 그리는 도구인 drawing_utils를 mp_draw라는 이름으로 선언
mp_draw = mp.solutions.drawing_utils 

holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 실시간 상태 관리를 위한 전역 변수
sequence = []           # 30프레임 데이터를 담는 바구니
sentence_list = []      # 인식된 단어들을 모은 리스트
last_translated = ""    # Gemini가 번역한 최종 문장

# ==========================================
# 3. 핵심 유틸리티 함수 (좌표 추출 및 번역)
# ==========================================

def extract_keypoints(results):
    """MediaPipe 결과에서 177개의 상대 좌표 추출"""
    # Pose: 얼굴 위치(0~10) + 상체(11~16) = 17개 점 사용
    if results.pose_landmarks:
        pose = np.array([[results.pose_landmarks.landmark[i].x, 
                          results.pose_landmarks.landmark[i].y, 
                          results.pose_landmarks.landmark[i].z] for i in range(17)]).flatten()
    else:
        pose = np.zeros(17*3)

    # Left Hand: 손목(0번) 기준 상대 좌표 (정규화)
    if results.left_hand_landmarks:
        lh_base = results.left_hand_landmarks.landmark[0]
        lh = np.array([[res.x - lh_base.x, res.y - lh_base.y, res.z - lh_base.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)
        
    # Right Hand: 손목(0번) 기준 상대 좌표 (정규화)
    if results.right_hand_landmarks:
        rh_base = results.right_hand_landmarks.landmark[0]
        rh = np.array([[res.x - rh_base.x, res.y - rh_base.y, res.z - rh_base.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
    
    return np.concatenate([pose, lh, rh])

async def get_gemini_translation(words):
    """단어 리스트를 전문 수어 통역사처럼 자연스러운 문장으로 변환"""
    if not words: return ""
    
     # [핵심] Gemini에게 부여하는 역할과 규칙을 상세하게 설정합니다.
    prompt = (
        f"당신은 농인과 청인을 연결하는 '전문 수어 통역사'입니다. "
        f"수어는 조사와 어미가 생략된 단어 위주의 나열로 이루어져 있습니다. "
        f"다음 나열된 수어 단어들의 맥락을 파악하여, 한국어 문법에 맞는 자연스럽고 정중한 문장으로 의역해 주세요.\n\n"
        f"수어 단어 목록: {' '.join(words)}\n\n"
        f"⚠️ 규칙:\n"
        f"1. '나 + 너'와 같이 중복되거나 나열된 주어는 '우리' 혹은 '저와 당신'으로 자연스럽게 통합하세요.\n"
        f"2. 전체적인 상황을 고려하여 적절한 조사와 서술어 어미(예: ~입니다, ~해요)를 붙이세요.\n"
        f"3. 불필요한 단어 반복은 삭제하고 핵심 의미만 전달하세요.\n"
        f"4. **다른 설명 없이 오직 완성된 문장 딱 한 줄만 출력하세요.**"
    )
    
    try:
        loop = asyncio.get_event_loop()
        # 스트리밍 방식이 아닌 일반 생성 방식으로 호출
        response = await loop.run_in_executor(None, lambda: gemini_model.generate_content(prompt))
        
        translated_text = response.text.strip()
        
        # 따옴표나 불필요한 기호 제거
        translated_text = translated_text.replace('"', '').replace("'", "")
        
        print(f"🤖 Gemini 번역 완료: {translated_text}") 
        return translated_text
    except Exception as e:
        print(f"❌ Gemini 에러: {e}")
        return " + ".join(words) # 에러 발생 시 단어 나열로 복구

# ==========================================
# 4. 웹소켓 통신 (리액트 실시간 연동)
# ==========================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global sequence, sentence_list, last_translated
    await websocket.accept()
    
    last_add_time = 0      
    idle_start_time = 0    
    
    cap = cv2.VideoCapture(0) 
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1) # 좌우 반전
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            # ------------------------------------------------------
            # [추가] 뼈대 그리기 (이미지 인코딩 전에 실행)
            # ------------------------------------------------------
            # 1. 포즈(몸) 뼈대 그리기
            if results.pose_landmarks:
                mp_draw.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4), # 관절 색상
                mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)  # 연결선 색상
                )

            # 2. 양손 뼈대 그리기
            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # 3. 얼굴 망(Face Mesh) 그리기 (너무 복잡해 보이면 이 부분만 주석 처리하세요)
            # if results.face_landmarks:
            #     mp_draw.draw_landmarks(
            #         frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            #         mp_draw.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            #         mp_draw.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
            #     )
            # ------------------------------------------------------

            # 좌표 추출 및 30프레임 데이터 업데이트
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:] 
            
            current_word = "IDLE"
            confidence = 0.0
            now = time.time()
            
            if len(sequence) == 30:
                if not results.left_hand_landmarks and not results.right_hand_landmarks:
                    current_word = "IDLE"
                    if idle_start_time == 0: idle_start_time = now
                    elif now - idle_start_time > 3.0: 
                        sentence_list = []
                        last_translated = ""
                else:
                    idle_start_time = 0 
                    res = model(np.expand_dims(sequence, axis=0), training=False)[0].numpy()
                    prediction_idx = np.argmax(res)
                    
                    if prediction_idx < len(actions):
                        current_word = actions[prediction_idx]
                        confidence = float(res[prediction_idx])
                        
                        if confidence > 0.98 and current_word != "IDLE" and (now - last_add_time > 2.0):
                            if not sentence_list or current_word != sentence_list[-1]:
                                sentence_list.append(current_word)
                                last_add_time = now
                                last_translated = await get_gemini_translation(sentence_list)

            display_word = current_word if (current_word != "IDLE" and confidence > 0.85) else "..."
            
            # 이미 뼈대가 그려진 frame을 인코딩해서 보냅니다.
            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            result_json = {
                "word": display_word,
                "confidence": confidence,
                "sentence": " + ".join(sentence_list) if sentence_list else "수어를 기다리는 중...",
                "translated": last_translated if last_translated else "문장을 만드는 중...",
                "image": jpg_as_text
            }
            
            await websocket.send_text(json.dumps(result_json))
            await asyncio.sleep(0.01)

    except Exception as e:
        print(f"❌ 웹소켓 연결 종료: {e}")
    finally:
        cap.release()
        await websocket.close()

# ==========================================
# 5. 서버 실행 (Uvicorn)
# ==========================================
if __name__ == "__main__":
    import uvicorn
    # 리액트 앱과 통신할 8080 포트로 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8080)