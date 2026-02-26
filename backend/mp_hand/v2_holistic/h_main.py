import cv2
import mediapipe as mp
import numpy as np
import os
import base64
import json
import asyncio
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import google.generativeai as genai
from dotenv import load_dotenv
import ollama

# .env 파일 로드
load_dotenv()

# ==========================================
# 1. 초기 설정 및 Gemini AI 세팅
# ==========================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gemini AI API 설정 ---
api_key = os.getenv("GOOGLE_API_KEY")
gemini_model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-flash-latest') 
        
        print("✅ Gemini API 연결 성공! (사용 모델: gemini-flash-latest)")
    except Exception as e:
        print(f"❌ Gemini 설정 실패: {e}")
else:
    print("❌ 에러: .env 파일에서 GOOGLE_API_KEY를 읽지 못했습니다.")

# ==========================================
# 2. 모델 및 수어 데이터 로드
# ==========================================
DATA_PATH = os.path.join('data')
# 폴더명을 가나다순으로 읽어와 학습 시 라벨링 순서와 일치시킴
actions = np.array(sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]))
print(f"✅ 인식 가능한 단어 리스트 ({len(actions)}개): {actions}")

# 학습된 LSTM 모델 로드
model = load_model(os.path.join('models', 'holistic_model.h5'))

# MediaPipe Holistic 설정 (complexity=0으로 성능 최적화)
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils 
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    model_complexity=0
)

# --- 실시간 상태 관리를 위한 전역 변수 ---
sequence = []           
sentence_list = []      
last_translated = ""    
is_translating = False  # 번역 중복 호출 방지용 깃발
last_gemini_call_time = 0 # 429 에러 방지용 쿨타임 변수

# ==========================================
# 3. 핵심 유틸리티 함수 (좌표 추출 및 번역)
# ==========================================

def extract_keypoints(results):
    """177개의 상대 좌표 추출 (Pose 17개 + Hands 21개씩)"""
    if results.pose_landmarks:
        pose = np.array([[results.pose_landmarks.landmark[i].x, 
                          results.pose_landmarks.landmark[i].y, 
                          results.pose_landmarks.landmark[i].z] for i in range(17)]).flatten()
    else:
        pose = np.zeros(17*3)

    if results.left_hand_landmarks:
        lh_base = results.left_hand_landmarks.landmark[0]
        lh = np.array([[res.x - lh_base.x, res.y - lh_base.y, res.z - lh_base.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)
        
    if results.right_hand_landmarks:
        rh_base = results.right_hand_landmarks.landmark[0]
        rh = np.array([[res.x - rh_base.x, res.y - rh_base.y, res.z - rh_base.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
    
    return np.concatenate([pose, lh, rh])

# async def get_gemini_translation(word_list):
#     """단어 리스트를 자연스러운 한국어 문장으로 의역"""
#     global gemini_model
#     if not gemini_model or not word_list: return ""
    
#     prompt = (
#         f"당신은 전문 수어 통역사입니다. 다음 나열된 수어 단어들 [{', '.join(word_list)}]를 "
#         f"맥락에 맞는 자연스럽고 정중한 한국어 문장으로 만들어주세요. "
#         f"다른 설명 없이 오직 완성된 문장 딱 한 줄만 말해줘."
#     )
    
#     try:
#         # 비동기 처리를 위해 스레드에서 실행
#         response = await asyncio.to_thread(gemini_model.generate_content, prompt)
#         translated_text = response.text.strip().replace('"', '').replace("'", "")
#         print(f"🤖 Gemini 통역: {translated_text}")
#         return translated_text
#     except Exception as e:
#         print(f"⚠️ Gemini 에러: {e}")
#         return " + ".join(word_list)

async def update_translation_task(word_list):
    """배경에서 번역을 수행하고 결과 변수를 업데이트 (카메라 끊김 방지)"""
    global last_translated, is_translating
    is_translating = True
    result = await get_llama_translation(word_list)
    last_translated = result
    is_translating = False
    
# [수정] 내 컴퓨터의 Llama 3를 사용하는 번역 함수
async def get_llama_translation(word_list):
    """로컬 RTX 4070의 Llama 3.2로 자연스러운 문장 생성"""
    if not word_list: return ""
    
    input_text = ", ".join(word_list)
    # 인공지능에게 줄 명령(프롬프트)
    prompt = (
        f"너는 전문 수어 통역사야. 아래 단어들을 조합해 '하나'의 자연스러운 한국어 문장을 만들어.\n"
        f"입력 단어: [{input_text}]\n\n"
        f"⚠️ 규칙(반드시 지킬 것):\n"
        f"1. 다른 설명, 인사말, 서론을 절대 하지 마.\n"
        f"2. 1., 2. 처럼 번호를 매기지 마.\n"
        f"3. 일본어나 영어 등 다른 언어를 섞지 말고 오직 한국어만 사용해.\n"
        f"4. 오직 완성된 문장 딱 한 줄만 출력해.\n"
        f"5. 입력되지 않은 정보를 마음대로 추가하거나 지어내지 마. 제공된 단어들의 의미 안에서만 문장을 연결해."
    )
    
    try:
        # RTX 4070의 힘을 빌려 문장 생성 (비차단 방식)
        response = await asyncio.to_thread(
            ollama.chat, 
            model='gemma3:4b', 
           messages=[
                # 시스템 역할을 부여하여 지능을 고정시킵니다.
                # {'role': 'system', 'content': '너는 오직 한국어로만 대답하는 전문 수어 통역사야. 절대 부연설명을 하지 마.'},
                {'role': 'system', 'content': '단어를 가지고 한줄로 요약한다. 되묻지 않는다.'},
                # {'role': 'user', 'content': prompt}
                {'role': 'user', 'content': "prompt"}
            ],
            # 생성 옵션을 조절하여 '창의성'을 낮추고 '정확도'를 높입니다.
            options={
                'temperature': 0.1, # 0에 가까울수록 정답만 말함
                'top_p': 0.9
            }
        )
        translated_text = response['message']['content'].strip()
        
        # 만약 결과에 따옴표가 포함되어 오면 제거
        translated_text = translated_text.replace('"', '').replace("'", "")
        
        print(f"🦙 Llama 3 통역: {translated_text}")
        return translated_text
        
    except Exception as e:
        print(f"❌ Llama 3 에러: {e}")
        return " + ".join(word_list)

# ==========================================
# 4. 웹소켓 통신 (메인 루프)
# ==========================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global sequence, sentence_list, last_translated, is_translating, last_gemini_call_time
    await websocket.accept()
    
    last_add_time = 0      
    idle_start_time = 0    
    frame_count = 0  
    
    curr_word = "..."
    curr_conf = 0.0

    cap = cv2.VideoCapture(0) 
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1) # 좌우 반전
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            
            # [시각화] 뼈대 그리기
            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # 좌표 추출 및 시퀀스 업데이트
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:] 
            
            now = time.time()
            frame_count += 1

            # [최적화] 5프레임마다 단어 예측 수행 (부하 감소)
            if frame_count % 5 == 0 and len(sequence) == 30:
                is_hands_up = False
                if results.pose_landmarks:
                    # 15, 16: 손목 / 23, 24: 골반(Hip)
                    left_wrist_y = results.pose_landmarks.landmark[15].y
                    right_wrist_y = results.pose_landmarks.landmark[16].y
                    hip_y = (results.pose_landmarks.landmark[23].y + results.pose_landmarks.landmark[24].y) / 2
                    # 배꼽 근처(골반 살짝 위)보다 손목이 높아야 '손 올림'으로 간주
                    if left_wrist_y < (hip_y - 0.05) or right_wrist_y < (hip_y - 0.05):
                        is_hands_up = True

                if not is_hands_up or (not results.left_hand_landmarks and not results.right_hand_landmarks):
                    curr_word = "IDLE"
                    curr_conf = 0.0
                    if idle_start_time == 0: idle_start_time = now
                    elif now - idle_start_time > 3.0: # 3초간 손 내리면 초기화
                        sentence_list = []
                        last_translated = ""
                else:
                    idle_start_time = 0 
                    # LSTM 모델 예측
                    res = model(np.expand_dims(sequence, axis=0), training=False)[0].numpy()
                    prediction_idx = np.argmax(res)
                    
                    if prediction_idx < len(actions):
                        curr_word = actions[prediction_idx]
                        curr_conf = float(res[prediction_idx])
                        
                        # --- [단어 추가 로직] ---
                        if curr_conf > 0.98 and curr_word != "IDLE" and (now - last_add_time > 3.0):
                            if curr_word not in sentence_list:
                                sentence_list.append(curr_word)
                                last_add_time = now
                                
                                # 단어가 딱 1개일 때는 AI 안 부르고 바로 표시 (속도 향상)
                                if len(sentence_list) == 1:
                                    last_translated = curr_word
                                # 2개 이상일 때만 Llama 3 호출
                                elif len(sentence_list) >= 2:
                                    if (now - last_gemini_call_time > 3.0) and not is_translating:
                                        asyncio.create_task(update_translation_task(sentence_list))
                                        last_gemini_call_time = now 
                                    else:
                                        # 번역 대기 중에는 단어들을 나열해서 미리 보여줌
                                        last_translated = " + ".join(sentence_list)

                # --- [초기화 로직 보강] ---
                # 손을 내리고 3초가 지나면 문장 리스트와 번역 결과 모두 삭제
                if not is_hands_up and idle_start_time != 0 and (now - idle_start_time > 3.0):
                    sentence_list = []
                    last_translated = "" # [추가] 화면을 깨끗하게 비움

            # --- [리액트 전송 준비] ---
            # 화면 중앙의 큰 글자: 90% 이상 확신할 때만 표시
            display_word = curr_word if (curr_word != "IDLE" and curr_conf > 0.90) else "..."
            
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            await websocket.send_text(json.dumps({
                "word": display_word,
                "confidence": curr_conf,
                "sentence": " + ".join(sentence_list) if sentence_list else "수어를 대기 중입니다...",
                "translated": last_translated,
                "image": jpg_as_text
            }))
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("ℹ️ 리액트 연결 종료")
    except Exception as e:
        print(f"❌ 에러: {e}")
    finally:
        cap.release()
        try: await websocket.close()
        except: pass

# ==========================================
# 5. 서버 실행
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)