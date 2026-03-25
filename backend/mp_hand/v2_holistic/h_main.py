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
# 우리가 이전에 열심히 모은 데이터로 학습시켜둔 AI의 뇌(holistic_model.h5)를 불러옴.
# (LSTM은 시간의 흐름, 즉 '동작의 연속'을 기억하는 데 특화된 AI 모델)
model = load_model(os.path.join('models', 'holistic_model.h5'))

# MediaPipe Holistic 설정 (complexity=0으로 성능 최적화)
mp_holistic = mp.solutions.holistic     # 얼굴, 몸통, 손을 한 번에 다 찾아주는 기능
mp_draw = mp.solutions.drawing_utils    # 찾은 뼈대 위에 선을 그려주는 기능(시각화) 
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,     # 추적할 때도 50%의 확신만 있으면 됨
    model_complexity=0               # [핵심] 0, 1, 2 중 0번. 정확도를 살짝 낮추는 대신 속도를 엄청 빠르게 만듭니다. (실시간 카메라에 필수!)
)

# --- 실시간 상태 관리를 위한 전역 변수 ---
sequence = []           # 최근 30장의 사진(프레임) 속 뼈대 좌표를 담아둘 바구니 (동작 인식용)
sentence_list = []      # AI가 인식한 수어 단어들을 차곡차곡 모아둘 문장 바구니
last_translated = []    # 엑사원 리스트 반환에 맞춰 빈 리스트로 초기화
is_translating = False  # 번역 중복 호출 방지용 깃발
last_gemini_call_time = 0 # 429 에러 방지용 쿨타임 변수

# ==========================================
# 3. 핵심 유틸리티 함수 (좌표 추출 및 번역)
# ==========================================

#사람의 뼈대를 숫자로 바꾸는 함수 (extract_keypoints)
def extract_keypoints(results):
    # 1. 몸통 좌표 추출 (어깨, 팔꿈치, 손목 등 17개 점)
    if results.pose_landmarks:
        pose_pts = results.pose_landmarks.landmark
        # 몸통 점들의 x, y, z 좌표를 하나로 쭉 이어서(flatten) 리스트로 만든다.
        pose = np.array([[pose_pts[i].x, pose_pts[i].y, pose_pts[i].z] for i in range(17)]).flatten()
        
        # [핵심] 카메라가 인식한 '내 몸의 손목 위치' (15번=왼손목, 16번=오른손목)
        pose_left_wrist = pose_pts[15]
        pose_right_wrist = pose_pts[16]
    else:
        # 몸이 안 보이면 전부 0으로 채움.
        pose = np.zeros(17*3)
        pose_left_wrist = pose_right_wrist = None

    # --- 손 좌표 유효성 검사 함수 (진짜 내 손이 맞는지 검사하는 탐정 역할) ---
    def get_valid_hand(hand_landmarks, pose_wrist):
        # 손이나 몸통이 안 보이면 0으로 반환
        if not hand_landmarks or not pose_wrist:
            return np.zeros(21*3)
        
        # 카메라가 찾은 '손 모양'의 손목(0번) 위치
        hand_wrist = hand_landmarks.landmark[0]
        
        # 내 몸의 손목과 카메라가 찾은 손목 사이의 거리를 계산. (점과 점 사이의 거리 공식)
        distance = np.sqrt((hand_wrist.x - pose_wrist.x)**2 + (hand_wrist.y - pose_wrist.y)**2)
        
        # [검증] 거리가 0.15(화면의 15%)보다 멀면? 
        # -> "내 팔끝에 달린 손이 아니라, 옆에 지나가는 다른 사람의 손이구나!" 하고 무시.
        if distance > 0.15:
            print("⚠️ 타인의 손 감지됨 - 무시합니다.") 
            return np.zeros(21*3)
        
        # 내 손이 맞다면, 손목을 기준점(0,0)으로 잡고 나머지 손가락 관절들의 위치를 계산해 반환합니다.
        return np.array([[res.x - hand_wrist.x, res.y - hand_wrist.y, res.z - hand_wrist.z] 
                         for res in hand_landmarks.landmark]).flatten()

    # 2. 왼손 추출 (다른 사람 손인지 검사 거침)
    lh = get_valid_hand(results.left_hand_landmarks, pose_left_wrist)
    
    # 3. 오른손 추출 (다른 사람 손인지 검사 거침)
    rh = get_valid_hand(results.right_hand_landmarks, pose_right_wrist)
    
    # 몸통, 왼손, 오른손 좌표를 하나의 긴 숫자 기차(배열)로 연결해서 반환.
    return np.concatenate([pose, lh, rh])

#투박한 수어 단어를 매끄러운 문장으로 바꾸는 함수 (get_exaone_translation)
# [수정] EXAONE 3.5를 사용하는 번역 함수
async def get_exaone_translation(word_list):
    """로컬 RTX 4070의 EXAONE 3.5로 자연스러운 문장 생성"""
    if not word_list: return [] # 단어가 없으면 그냥 돌아감
    
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
        # 컴퓨터에 깔려있는 AI(Ollama - exaone3.5)에게 번역을 부탁.
        # asyncio.to_thread: 이 작업이 오래 걸려도 카메라 화면이 멈추지 않게 '백그라운드'에서 일하게 만든다.
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
        # 1. AI가 준 전체 텍스트 가져오기,앞뒤 공백 자름.
        full_content = response['message']['content'].strip()
        
        # 2. 줄바꿈(\n) 기준으로 쪼개서 리스트로 만들기
        lines = full_content.split('\n')
        translated_options = []
        
        for line in lines:
            if not line.strip(): continue   # 빈 줄은 무시
            
            # [강력 필터링] 정규표현식을 사용해 "Option 1:", "01.", "정중한 말투:" 등을 강제로 지웁니다.
            # re.sub: '특정 패턴의 글자를 찾아서 지워버려라'는 마법의 지우개
            # 1. 'Option 1', '옵션 1', '1.' 등 숫자와 기호 제거
            # 2. 콜론(:)이나 괄호()) 뒤의 내용만 추출
            clean_line = re.sub(r'^(Option\s*\d+|옵션\s*\d+|[\d\.]+|[-*])[:\s\)]*', '', line, flags=re.IGNORECASE).strip()
            # 3. 혹시나 남은 "격식:", "친근:" 같은 단어도 제거
            clean_line = re.sub(r'^.*?(말투|격식|친근|심플|간결|요약|인사|스타일)[:\s]+', '', clean_line).strip()
            
            if clean_line:  # 깨끗해진 문장만 리스트에 담는다.
                translated_options.append(clean_line.replace('"', '').replace("'", ""))
        
        # 최종적으로 위에서부터 3개의 문장만 딱 잘라서 반환
        final_result = translated_options[:3]
        print(f"🇰🇷 EXAONE 최종 가공 결과: {final_result}")
        return final_result
        
    except Exception as e:
        print(f"❌ EXAONE 에러: {e}")
        return [" + ".join(word_list)]  # 에러가 나면 그냥 단어들을 '+'로 연결해서 임시로 보여줌
    
    
 #번역 작업을 뒷배경에서 조용히 처리하는 함수 (update_translation_task)   
async def update_translation_task(word_list):
    """배경에서 번역을 수행하고 결과 변수를 업데이트 (카메라 끊김 방지)"""
    # 전역 변수(외부에 있는 메모장)를 여기서 수정하겠다고 선언
    global last_translated, is_translating
    
    # 만약 이미 번역 중이라면? 새 주문을 받지 않고 그냥 무시합니다. (AI 과부하 방지)
    if is_translating: return 
    
    # "이제 번역 시작합니다!" 하고 깃발을 꽂습니다.
    is_translating = True
    
    # 아까 만든 번역 함수를 부릅니다.
    result = await get_exaone_translation(word_list)
    
    # 번역이 완료되면 결과를 메모장(last_translated)에 적습니다.
    last_translated = result
    
    # "번역 끝났습니다!" 하고 깃발을 내립니다.
    is_translating = False

# ==========================================
# 4. 웹소켓 통신 (메인 루프)
# ==========================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # 1. 전역 변수 선언 (외부 변수를 수정하기 위해 필요)
    global sequence, sentence_list, last_translated, is_translating, last_gemini_call_time
    
    # 2. 웹소켓 연결 수락(프론트엔드(React)에서 온 연결을 받는다.)
    await websocket.accept()
    print("✅ 리액트 연결됨 - 이전 데이터를 초기화합니다.")
    
    
    # 3. [핵심] 기존에 남아있던 데이터 싹 비우기 (리셋 로직)
    sequence = []           # 30프레임 바구니 초기화
    sentence_list = []      # 인식된 단어 목록 초기화
    last_translated = []    # 엑사원이 만든 문장 후보들 초기화 (리스트 형태)
    is_translating = False  # 번역 중인 상태 해제
    last_gemini_call_time = 0 # 쿨타임 타이머 리셋

    # 4. 루프 내에서 사용할 로컬 변수(도구들 준비) 초기화
    last_add_time = 0      # 마지막으로 단어를 추가한 시간 (너무 빨리 중복 추가되는 것 방지)
    idle_start_time = 0    # 손을 내리고 쉰 시간 체크용
    frame_count = 0        # 카메라 프레임(사진) 번호
    curr_word = "..."      # 현재 화면에 띄울 단어
    curr_conf = 0.0        # AI가 이 단어를 얼마나 확신하는지 (0~100%)

    # 5. 카메라 실행 (이후 로직은 동일,0번은 내장 웹캠을 의미)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다. 인덱스 확인 혹은 다른 앱 종료가 필요합니다.")
        await websocket.close()
        return

    try:
        # [메인 루프] 카메라가 켜져 있는 동안 무한 반복되는 컨베이어 벨트
        while cap.isOpened():
            # 1. 카메라에서 1장의 사진(frame)을 찍어온다
            ret, frame = cap.read()
            if not ret: break   # 사진 찍기 실패하면 중지
            
            frame = cv2.flip(frame, 1) # 거울처럼 보이게 좌우 반전
            # MediaPipe(뼈대 찾는 도구)가 좋아하는 색상 형식으로 바꾸기
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
             # 2. 사진 속에서 사람 뼈대 찾기
            results = holistic.process(image_rgb)
            
            # [시각화] 뼈대 그리기
            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # 3. 뼈대 좌표 추출 및 시퀀스(sequence) 업데이트
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            # [중요] 수어는 사진 1장이 아니라 '움직임'. 최근 30장의 사진(약 1초 분량의 움짤)만 유지
            sequence = sequence[-30:] 
            
            now = time.time()
            frame_count += 1

            # 4. [최적화] 5프레임마다 단어 예측 수행 (부하 감소)
            if frame_count % 5 == 0 and len(sequence) == 30:
                is_hands_up = False
                
                # [스탠바이 확인] 손목(15, 16)이 골반(23, 24)보다 위에 있는지 확인
                # "손을 올려야만 수어를 시작한 것으로 간주하겠다!" 라는 로직                
                if results.pose_landmarks:
                    # 15, 16: 손목 / 23, 24: 골반(Hip)
                    left_wrist_y = results.pose_landmarks.landmark[15].y
                    right_wrist_y = results.pose_landmarks.landmark[16].y
                    hip_y = (results.pose_landmarks.landmark[23].y + results.pose_landmarks.landmark[24].y) / 2
                    # 배꼽 근처(골반 살짝 위)보다 손목이 높아야 '손 올림'으로 간주
                    if left_wrist_y < (hip_y - 0.05) or right_wrist_y < (hip_y - 0.05):
                        is_hands_up = True

                # 손을 내리고 있거나, 화면에 손이 아예 안 보이면? -> 휴식 상태(IDLE)
                if not is_hands_up or (not results.left_hand_landmarks and not results.right_hand_landmarks):
                    curr_word = "IDLE"
                    curr_conf = 0.0
                    if idle_start_time == 0: idle_start_time = now
                    elif now - idle_start_time > 10.0: # # 손 내리고 10초 지나면 초기화
                        sentence_list = []
                        last_translated = []
                else:
                     # 손을 다시 올리면 타이머가 리셋(초기화X)
                    idle_start_time = 0     
                    
                    # LSTM 모델 예측
                    # 30장의 사진 묶음(움짤)을 LSTM 모델(우리가 학습시킨 AI)에게 보여주고 정답을 요구
                    res = model(np.expand_dims(sequence, axis=0), training=False)[0].numpy()
                    
                    # AI가 가장 자신있어 하는 단어의 번호를 찾는다.
                    prediction_idx = np.argmax(res)
                    
                    if prediction_idx < len(actions):
                        curr_word = actions[prediction_idx] # 예: "안녕하세요"
                        curr_conf = float(res[prediction_idx]) # 확신도 예: 0.99 (99%)
                        
                        # --- [단어 추가 로직] ---
                        # 98% 이상 확신하고, 쉬는 상태(IDLE)가 아니며, 이전 단어를 넣은 지 3초가 지났다면?
                        if curr_conf > 0.98 and curr_word != "IDLE" and (now - last_add_time > 3.0):
                            if curr_word not in sentence_list:
                                sentence_list.append(curr_word) # 단어장에 단어 추가
                                last_add_time = now
                                
                                # 1단계: 단어가 딱 1개일 때는 AI 안 부르고 바로 표시 (속도 향상)
                                if len(sentence_list) == 1:
                                    last_translated = [curr_word]
                                    
                                # 2단계: 2개 이상일 때만 EXAONE 호출
                                elif len(sentence_list) >= 2:
                                    # 단, 3초에 1번씩만 부탁해서 렉 걸리는 걸 막습니다.
                                    if (now - last_gemini_call_time > 3.0) and not is_translating:
                                        # 비동기로 조용히 뒷배경에서 번역을 시작
                                        asyncio.create_task(update_translation_task(sentence_list))
                                        last_gemini_call_time = now 
                                    else:
                                        # 번역 대기 중에는 임시로 "단어 + 단어" 형태로 나열해서 미리 보여줌
                                        last_translated = " + ".join(sentence_list)

                # --- [초기화 로직 보강] ---
                # 손을 내리고 3초가 지나면 문장 리스트와 번역 결과 모두 삭제
                if not is_hands_up and idle_start_time != 0 and (now - idle_start_time > 3.0):
                    sentence_list = []
                    last_translated = "" # [추가] 화면을 깨끗하게 비움

            # --- [리액트 전송 준비] ---
            # 화면 중앙의 큰 글자: 90% 이상 확신할 때만 표시
            display_word = curr_word if (curr_word != "IDLE" and curr_conf > 0.90) else "..."
            
            # [마법의 기술] 카메라 사진(이미지)을 텍스트(Base64)로 변환
            # 왜냐하면 웹소켓은 글자만 주고받기 편하기 때문에, 사진을 엄청나게 긴 글자로 압축해서 보내는 것
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            # 프론트엔드로 JSON 데이터 보내기
            await websocket.send_text(json.dumps({
                "word": display_word,                       # 지금 인식한 단어
                "confidence": curr_conf,                    # 확신도
                "sentence": " + ".join(sentence_list) if sentence_list else "수어를 대기 중입니다...",  # 나열된 단어들
                "translated": last_translated,              # AI가 번역해 준 정리된 문장
                "image": jpg_as_text                        # 카메라 화면 (사진)
            }))
            
            # 너무 빨리 돌아서 컴퓨터 터지는 걸 방지 (0.01초 휴식)
            await asyncio.sleep(0.01)
            
    # 프론트엔드 창을 끄면 연결이 끊어짐
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
# 파이썬 파일을 직접 실행하면 웹 서버(Uvicorn)를 포트 8080번에서 열어줌
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)