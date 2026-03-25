import cv2
import mediapipe as mp
import numpy as np
import os
import time
from PIL import ImageFont, ImageDraw, Image

# ==========================================
# 1. 어떤 단어를 녹화할지 설정하는 곳
# ==========================================
action_name = "안녕하세요"  # 촬영할 단어 이름
max_len = 30               # 한 동작당 30프레임 (1초) 연속으로

# 데이터를 저장할 창고
data_path = os.path.join('data')
output_dir = os.path.join(data_path, action_name)
os.makedirs(output_dir, exist_ok=True)  # 폴더가 없으면 만들고, 있으면 그냥 둔다.

# ==========================================
# 2. 뼈대 그리기 도구 (MediaPipe) 세팅
# ==========================================
# --- MediaPipe 초기화 ---
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils # 점과 선을 그리는 도구

# 선의 스타일 설정 (BGR 색상 기준 - 색상, 두께 등)
pose_style = mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4) # 몸통 선
hand_style = mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2)  # 손 선


holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    model_complexity=0  # 화면이 안 끊기게 속도를 제일 빠르게(0) 설정
)

# ==========================================
# 3. 도우미 함수 1: 화면에 한글 띄우기
# ==========================================
# (참고: 기본 카메라 툴(OpenCV)은 한글을 지원하지 않아서 글자가 깨집니다.)
def draw_text(img, text, position, font_size=30, color=(255, 255, 255)):
    try:
        # 카메라 사진을 그림판(PIL)으로 잠깐 가져와서 '맑은 고딕(malgun.ttf)'으로 한글을 예쁘게 씁니다.
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype("malgun.ttf", font_size)
        draw.text(position, text, font=font, fill=color)
        
        # 글자를 다 썼으면 다시 카메라용 사진으로 되돌려줍니다.
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except:
        # 폰트가 없어서 에러가 나면 그냥 영어 폰트로 대충 씁니다.
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return img

# ==========================================
# 4. 도우미 함수 2: 사람 몸을 숫자로 바꾸기(좌표 추출)
# ==========================================
def extract_keypoints(results):
    """177개 좌표 추출 (몸통, 왼손, 오른손)"""
    
    # 1. 몸통 (17개 점 x 3차원 좌표 = 51개 숫자)
    if results.pose_landmarks:
        pose = np.array([[results.pose_landmarks.landmark[i].x, 
                          results.pose_landmarks.landmark[i].y, 
                          results.pose_landmarks.landmark[i].z] for i in range(17)]).flatten()
    else:
        pose = np.zeros(17*3) # 몸이 안 보이면 다 0으로 채움

    # 2. 왼손 (21개 점 x 3차원 좌표 = 63개 숫자)
    if results.left_hand_landmarks:
        lh_base = results.left_hand_landmarks.landmark[0] # 손목을 0번(기준점)으로 잡습니다.
        # [핵심] 모든 손가락 관절 위치에서 손목 위치를 뺍니다. (= '손목'을 중심으로 모양만 기억하게 만듭니다)
        lh = np.array([[res.x - lh_base.x, res.y - lh_base.y, res.z - lh_base.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)
        
    # 3. 오른손 (왼손과 동일)
    if results.right_hand_landmarks:
        rh_base = results.right_hand_landmarks.landmark[0]
        rh = np.array([[res.x - rh_base.x, res.y - rh_base.y, res.z - rh_base.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
    
    # 이 세 덩어리를 하나의 긴 숫자 기차(총 177개 숫자)로 합쳐서 반환합니다.
    return np.concatenate([pose, lh, rh])

# ==========================================
# 5. 메인 루프 (카메라 켜기)
# ==========================================
# 촬영 대기 모드 (거울 보기)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)    # 카메라 ON

print(f"🚀 [{action_name}] 촬영 모드 시작")

while cap.isOpened():
    ret, frame = cap.read() # 1장 찍기
    if not ret: break
    frame = cv2.flip(frame, 1)  # 거울 모드 (좌우 반전)

    # 실시간 뼈대 추출 (촬영 전 가이드용)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    # --- [핵심] 화면에 뼈대 그리기 ---
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, pose_style, pose_style)
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, hand_style, hand_style)
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, hand_style, hand_style)

    # 안내 문구 (한글 띄우기 함수 사용)
    frame = draw_text(frame, f"단어: {action_name}", (10, 30), font_size=35, color=(255, 255, 0))
    frame = draw_text(frame, "'S'키: 촬영 시작 | 'Q'키: 종료", (10, 80), font_size=20)
    
    # 내 모습을 모니터(창)에 띄웁니다.
    cv2.imshow('Holistic Data Collector', frame)
    
    # 키보드 입력을 0.001초 동안 기다립니다.
    key = cv2.waitKey(1)
    
    # ==========================================
    # 진짜 녹화 시작 ('S'키를 눌렀을 때)
    # ==========================================
    if key == ord('s'):
        # 1. 2초 카운트다운 루프
        for countdown in range(2, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            
            # 카운트다운 중에도 뼈대 표시
            res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks: mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS, pose_style)
            
            # 화면 한가운데에 "준비! 2", "준비! 1" 글자를 띄웁니다.
            frame = draw_text(frame, f"준비! {countdown}", (200, 200), font_size=80, color=(0, 0, 255))
            cv2.imshow('Holistic Data Collector', frame)
            # 1000밀리초 = 딱 1초 동안 기다립니다.
            cv2.waitKey(1000)

        # 2. 진짜 녹화 루프 (30프레임)
        video_landmarks = []    # 30장의 숫자를 담을 빈 상자
        for i in range(max_len):    # 0부터 29까지 반복 (30번)
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # 녹화 중에도 뼈대 표시
            if results.pose_landmarks: mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, pose_style)
            if results.left_hand_landmarks: mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, hand_style)
            if results.right_hand_landmarks: mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, hand_style)

            # 데이터 추출 및 저장용 리스트 추가
             # [가장 중요] 지금 이 순간의 내 몸을 숫자로 뽑아냅니다. (177개 숫자)
            keypoints = extract_keypoints(results)
            # 빈 상자에 숫자들을 차곡차곡 넣습니다.
            video_landmarks.append(keypoints)
            
             # 화면 구석에 "RECORDING... 1/30" 식으로 녹화 진행 상황을 띄웁니다.
            cv2.putText(frame, f"RECORDING... {i+1}/{max_len}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Holistic Data Collector', frame)
            cv2.waitKey(1)

        # 3. 파일 저장
        # 폴더 안에 이미 파일이 몇 개 있는지 세어봅니다. (0.npy, 1.npy 식으로 이름 짓기 위해)
        file_idx = len([f for f in os.listdir(output_dir) if f.endswith('.npy')])
        # 30장의 숫자가 담긴 상자를 '.npy'라는 데이터 파일로 저장
        np.save(os.path.join(output_dir, f"{file_idx}.npy"), np.array(video_landmarks))
        print(f"✅ 저장 완료: {action_name}_{file_idx}.npy")

# 'q' 키를 누르면 프로그램 종료
    elif key == ord('q'):
        break
    
# 카메라 전원 끄기
cap.release()
cv2.destroyAllWindows()