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
import re
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
last_translated = []    # 엑사원 리스트 반환에 맞춰 빈 리스트로 초기화
is_translating = False  # 번역 중복 호출 방지용 깃발
last_gemini_call_time = 0 # 429 에러 방지용 쿨타임 변수

# ==========================================
# 3. 핵심 유틸리티 함수 (좌표 추출 및 번역)
# ==========================================

def extract_keypoints(results):
     # 1. 포즈 좌표 추출 (어깨, 팔꿈치, 손목 등 17개 점)
    if results.pose_landmarks:
        pose_pts = results.pose_landmarks.landmark
        pose = np.array([[pose_pts[i].x, pose_pts[i].y, pose_pts[i].z] for i in range(17)]).flatten()
        
        # [핵심] 내 몸이 생각하는 손목의 위치 (주인공 기준점)
        pose_left_wrist = pose_pts[15]
        pose_right_wrist = pose_pts[16]
    else:
        pose = np.zeros(17*3)
        pose_left_wrist = pose_right_wrist = None

    # --- 손 좌표 유효성 검사 함수 ---
    def get_valid_hand(hand_landmarks, pose_wrist):
        if not hand_landmarks or not pose_wrist:
            return np.zeros(21*3)
        
        # 감지된 손의 손목(0번) 위치
        hand_wrist = hand_landmarks.landmark[0]
        
        # 내 몸의 손목(Pose 15/16)과 감지된 손의 손목(Hand 0) 사이의 거리 계산
        # 2D 거리 (x, y)만 체크해도 충분합니다.
        distance = np.sqrt((hand_wrist.x - pose_wrist.x)**2 + (hand_wrist.y - pose_wrist.y)**2)
        
        # [검증] 거리가 0.15(화면의 약 15%)보다 멀면 다른 사람의 손으로 간주하고 버림
        if distance > 0.15:
            print("⚠️ 타인의 손 감지됨 - 무시합니다.") # 디버깅용
            return np.zeros(21*3)
        
        # 내 손이 맞다면 상대 좌표(정규화)로 변환해서 반환
        return np.array([[res.x - hand_wrist.x, res.y - hand_wrist.y, res.z - hand_wrist.z] 
                         for res in hand_landmarks.landmark]).flatten()

    # 2. 왼손 추출 (검증 거침)
    lh = get_valid_hand(results.left_hand_landmarks, pose_left_wrist)
    
    # 3. 오른손 추출 (검증 거침)
    rh = get_valid_hand(results.right_hand_landmarks, pose_right_wrist)
    
    return np.concatenate([pose, lh, rh])

# [수정] EXAONE 3.5를 사용하는 번역 함수
async def get_exaone_translation(word_list):
    """로컬 RTX 4070의 EXAONE 3.5로 자연스러운 문장 생성"""
    if not word_list: return []
    
    input_text = ", ".join(word_list)
    # 인공지능에게 줄 명령(프롬프트)
    prompt = (
        f"너는 전문 수어 통역사야. 아래 나열된 단어들은 인식된 순서일 뿐이야. "
        f"단어의 순서에 집착하지 말고, 이 단어들이 가진 '핵심 의미'를 파악해서 "
        f"실제 한국 사람들이 일상 생활에서 대화할 때 쓰는 가장 자연스러운 문장으로 의역해줘.\n\n"
        f"입력된 단어 재료: [{input_text}]\n\n"
        f"⚠️ 필수 규칙:\n"
        f"1. 단어 나열 순서가 어색해도 무시하고, 문맥에 맞게 어순을 완전히 새로 배치해.\n"
        f"2. 비슷한 의미의 단어(예: 만나다, 반갑다)는 '만나서 반가워요'처럼 하나로 매끄럽게 합쳐.\n"
        f"3. 절대 단어를 기계적으로 나열하지 마. (예: '반갑다 만나다' -> '만나서 정말 반갑습니다' 식으로 의역)\n"
        f"4. 아래 3가지 스타일로 딱 한 줄씩만 출력해:\n"
        f"   - Option 1 (격식): 예의 바르고 정중한 사회적 인사말 (~습니다)\n"
        f"   - Option 2 (친근): 일상적인 대화에서 쓰는 다정한 말투 (~해요/네요)\n"
        f"   - Option 3 (심플): 불필요한 수식어를 빼고 핵심만 담은 깔끔한 문장\n"
        f"5. **중요: 'Option 1:', '정중한 말투:', '1.' 같은 모든 머리말은 무조건 삭제하고 '순수 문장 내용'만 출력해.**"
    )
    
    try:
        # RTX 4070의 힘을 빌려 문장 생성 (비차단 방식)
        response = await asyncio.to_thread(
            ollama.chat, 
            model='exaone3.5:7.8b', 
           messages=[
                # 시스템 역할을 부여하여 지능을 고정시킵니다.
                # {'role': 'system', 'content': '너는 오직 한국어로만 대답하는 전문 수어 통역사야. 절대 부연설명을 하지 마.'},
                {'role': 'system', 'content': '너는 불필요한 설명 없이 입력된 단어만으로 3가지 스타일의 문장을 만드는 전문 한국어 통역사야.'},
                # {'role': 'user', 'content': prompt}
                {'role': 'user', 'content': prompt}
            ],
            # 생성 옵션을 조절하여 '창의성'을 낮추고 '정확도'를 높입니다.
            options={
                'temperature': 0.3, # 0에 가까울수록 정답만 말함
                'top_p': 0.9
            }
        )
        # 1. AI가 준 전체 텍스트 가져오기
        full_content = response['message']['content'].strip()
        
        # 2. 줄바꿈(\n) 기준으로 쪼개서 리스트로 만들기
        lines = full_content.split('\n')
        translated_options = []
        for line in lines:
            if not line.strip(): continue
            
            # [강력 필터링] 정규표현식을 사용해 "Option 1:", "01.", "정중한 말투:" 등을 강제로 지웁니다.
            # 1. 'Option 1', '옵션 1', '1.' 등 숫자와 기호 제거
            # 2. 콜론(:)이나 괄호()) 뒤의 내용만 추출
            clean_line = re.sub(r'^(Option\s*\d+|옵션\s*\d+|[\d\.]+|[-*])[:\s\)]*', '', line, flags=re.IGNORECASE).strip()
            # 3. 혹시나 남은 "격식:", "친근:" 같은 단어도 제거
            clean_line = re.sub(r'^.*?(말투|격식|친근|심플|간결|요약|인사|스타일)[:\s]+', '', clean_line).strip()
            
            if clean_line:
                translated_options.append(clean_line.replace('"', '').replace("'", ""))
        
        final_result = translated_options[:3]
        print(f"🇰🇷 EXAONE 최종 가공 결과: {final_result}")
        return final_result
        
    except Exception as e:
        print(f"❌ EXAONE 에러: {e}")
        return [" + ".join(word_list)]
    
async def update_translation_task(word_list):
    """배경에서 번역을 수행하고 결과 변수를 업데이트 (카메라 끊김 방지)"""
    global last_translated, is_translating
    if is_translating: return 
    is_translating = True
    result = await get_exaone_translation(word_list)
    last_translated = result
    is_translating = False

# ==========================================
# 4. 웹소켓 통신 (메인 루프)
# ==========================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # 1. 전역 변수 선언 (외부 변수를 수정하기 위해 필요)
    global sequence, sentence_list, last_translated, is_translating, last_gemini_call_time
    
    # 2. 웹소켓 연결 수락
    await websocket.accept()
    print("✅ 리액트 연결됨 - 이전 데이터를 초기화합니다.")
    
    
    # 3. [핵심] 기존에 남아있던 데이터 싹 비우기 (리셋 로직)
    sequence = []           # 30프레임 바구니 초기화
    sentence_list = []      # 인식된 단어 목록 초기화
    last_translated = []    # 엑사원이 만든 문장 후보들 초기화 (리스트 형태)
    is_translating = False  # 번역 중인 상태 해제
    last_gemini_call_time = 0 # 쿨타임 타이머 리셋

    # 4. 루프 내에서 사용할 로컬 변수 초기화
    last_add_time = 0      
    idle_start_time = 0    
    frame_count = 0  
    curr_word = "..."
    curr_conf = 0.0

    # 5. 카메라 실행 (이후 로직은 동일)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다. 인덱스 확인 혹은 다른 앱 종료가 필요합니다.")
        await websocket.close()
        return

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
                    elif now - idle_start_time > 10.0: # 10초간 손 내리면 초기화
                        sentence_list = []
                        last_translated = []
                else:
                     # 손을 다시 올리면 타이머가 리셋되어 초기화되지 않습니다.
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
                                    last_translated = [curr_word]
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
        print("❌ 백엔드 로직 에러 발생!")
        
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