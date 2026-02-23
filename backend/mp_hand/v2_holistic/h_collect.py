import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image

# [설정] 저장 경로 및 단어 설정
# v2_holistic/data 폴더 안에 단어별로 저장되도록 설정
action_name = "반갑다"  # 촬영할 단어 이름으로 변경하세요
output_dir = os.path.join('data', action_name)
os.makedirs(output_dir, exist_ok=True)

max_len = 30  # Holistic은 데이터가 무거우므로 30프레임(약 1초) 추천

# --- 한글 출력을 위한 함수 (기존 코드 유지) ---
def draw_text(img, text, position, font_size=30, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("malgun.ttf", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- MediaPipe Holistic 핵심 좌표 추출 함수 ---
def extract_keypoints(results):
    # Pose: 33개 점 (전체 사용 혹은 슬라이싱)
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    
    # Face: 핵심 30개 점만 (다이어트 버전)
    FACE_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
    if results.face_landmarks:
        face = np.array([[results.face_landmarks.landmark[i].x, results.face_landmarks.landmark[i].y, results.face_landmarks.landmark[i].z] for i in FACE_LANDMARKS]).flatten()
    else:
        face = np.zeros(len(FACE_LANDMARKS)*3)
        
    # Hands: 좌/우 각 21개 점
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, face, lh, rh])

# MediaPipe Holistic 설정
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

print(f"🚀 [{action_name}] 수동 촬영 모드")
print("S키: 녹화 시작 | Q키: 종료")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    # 가이드 표시
    frame = draw_text(frame, f"단어: {action_name}", (10, 30), font_size=35, color=(255, 255, 0))
    frame = draw_text(frame, "'S'를 누르면 촬영 시작", (10, 80), font_size=20)
    
    cv2.imshow('Holistic Data Collector', frame)
    key = cv2.waitKey(1)
    
    # --- 's' 키 누르면 수집 시작 ---
    if key == ord('s'):
        video_landmarks = []
        print(f"🔴 녹화 중...")
        
        while len(video_landmarks) < max_len:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            # Holistic 처리
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # 뼈대 그리기
            mp_draw.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            # 좌표 추출
            keypoints = extract_keypoints(results)
            video_landmarks.append(keypoints)
            
            # 진행률 표시
            frame = draw_text(frame, f"녹화 중... {len(video_landmarks)}/{max_len}", (10, 30), color=(0, 0, 255))
            cv2.imshow('Holistic Data Collector', frame)
            cv2.waitKey(1)

        # --- 저장 (덮어쓰기 방지: 파일 개수 체크) ---
        file_idx = len([f for f in os.listdir(output_dir) if f.endswith('.npy')])
        save_path = os.path.join(output_dir, f"{file_idx}.npy")
        np.save(save_path, np.array(video_landmarks))
        print(f"✅ 저장 완료: {save_path} (총 {file_idx+1}개 데이터)")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()