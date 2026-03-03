import cv2
import mediapipe as mp
import numpy as np
import os
import time
from PIL import ImageFont, ImageDraw, Image

# [설정] 
action_name = "안녕하세요"  # 촬영할 단어 이름
max_len = 30               # 한 동작당 30프레임 (1초)
data_path = os.path.join('data')
output_dir = os.path.join(data_path, action_name)
os.makedirs(output_dir, exist_ok=True)

# --- MediaPipe 초기화 ---
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils # 점과 선을 그리는 도구
# 선의 스타일 설정 (색상, 두께 등)
pose_style = mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4)
hand_style = mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2)

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    model_complexity=0  # 시각화 반응 속도를 위해 0으로 설정
)

def draw_text(img, text, position, font_size=30, color=(255, 255, 255)):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype("malgun.ttf", font_size)
        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except:
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return img

def extract_keypoints(results):
    """177개 좌표 추출 (기존 규격 유지)"""
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

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print(f"🚀 [{action_name}] 촬영 모드 시작")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

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

    # 안내 문구
    frame = draw_text(frame, f"단어: {action_name}", (10, 30), font_size=35, color=(255, 255, 0))
    frame = draw_text(frame, "'S'키: 촬영 시작 | 'Q'키: 종료", (10, 80), font_size=20)
    
    cv2.imshow('Holistic Data Collector', frame)
    key = cv2.waitKey(1)
    
    if key == ord('s'):
        # 1. 2초 카운트다운 루프
        for countdown in range(2, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            # 카운트다운 중에도 뼈대 표시
            res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks: mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS, pose_style)
            
            frame = draw_text(frame, f"준비! {countdown}", (200, 200), font_size=80, color=(0, 0, 255))
            cv2.imshow('Holistic Data Collector', frame)
            cv2.waitKey(1000)

        # 2. 진짜 녹화 루프 (30프레임)
        video_landmarks = []
        for i in range(max_len):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # 녹화 중에도 뼈대 표시
            if results.pose_landmarks: mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, pose_style)
            if results.left_hand_landmarks: mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, hand_style)
            if results.right_hand_landmarks: mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, hand_style)

            # 데이터 추출 및 저장용 리스트 추가
            keypoints = extract_keypoints(results)
            video_landmarks.append(keypoints)
            
            cv2.putText(frame, f"RECORDING... {i+1}/{max_len}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Holistic Data Collector', frame)
            cv2.waitKey(1)

        # 3. 파일 저장
        file_idx = len([f for f in os.listdir(output_dir) if f.endswith('.npy')])
        np.save(os.path.join(output_dir, f"{file_idx}.npy"), np.array(video_landmarks))
        print(f"✅ 저장 완료: {action_name}_{file_idx}.npy")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()