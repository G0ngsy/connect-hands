import cv2
import mediapipe as mp
import numpy as np
import os
import time
from PIL import ImageFont, ImageDraw, Image

# [설정]
action_name = "반갑다"  # 단어 이름을 바꿔가며 촬영하세요
max_len = 30               # 한 동작당 30프레임 (1초)
data_path = os.path.join('data')
output_dir = os.path.join(data_path, action_name)
os.makedirs(output_dir, exist_ok=True)

# MediaPipe 초기화 (FaceMesh 제외하고 Pose만 사용)
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
    # Pose (0~16번 점: 얼굴 위치 + 상체) - 17개 점 * 3 = 51개 수치
    if results.pose_landmarks:
        pose = np.array([[results.pose_landmarks.landmark[i].x, 
                          results.pose_landmarks.landmark[i].y, 
                          results.pose_landmarks.landmark[i].z] for i in range(17)]).flatten()
    else:
        pose = np.zeros(17*3)

    # Left Hand (상대 좌표 - 손목 기준) - 21개 점 * 3 = 63개 수치
    if results.left_hand_landmarks:
        lh_base = results.left_hand_landmarks.landmark[0]
        lh = np.array([[res.x - lh_base.x, res.y - lh_base.y, res.z - lh_base.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)
        
    # Right Hand (상대 좌표 - 손목 기준) - 21개 점 * 3 = 63개 수치
    if results.right_hand_landmarks:
        rh_base = results.right_hand_landmarks.landmark[0]
        rh = np.array([[res.x - rh_base.x, res.y - rh_base.y, res.z - rh_base.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)
    
    # 총합: 51 + 63 + 63 = 177개 수치
    return np.concatenate([pose, lh, rh])

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    frame = draw_text(frame, f"단어: {action_name}", (10, 30), font_size=35, color=(255, 255, 0))
    frame = draw_text(frame, "'S'키를 누르면 2초 후 촬영", (10, 80), font_size=20)
    
    cv2.imshow('Holistic Collector', frame)
    key = cv2.waitKey(1)
    
    if key == ord('s'):
        # 2초 카운트다운 (준비 시간)
        for countdown in range(2, 0, -1):
            ret, frame = cap.read()
            temp_frame = cv2.flip(frame, 1)
            temp_frame = draw_text(temp_frame, f"준비! {countdown}", (200, 200), font_size=80, color=(0,0,255))
            cv2.imshow('Holistic Collector', temp_frame)
            cv2.waitKey(1000)

        video_landmarks = []
        for _ in range(max_len):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # 뼈대 그리기 (FaceMesh 제외)
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            
            keypoints = extract_keypoints(results)
            video_landmarks.append(keypoints)
            
            cv2.putText(frame, f"RECORDING... {len(video_landmarks)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Holistic Collector', frame)
            cv2.waitKey(1)

        file_idx = len([f for f in os.listdir(output_dir) if f.endswith('.npy')])
        np.save(os.path.join(output_dir, f"{file_idx}.npy"), np.array(video_landmarks))
        print(f"✅ 저장 완료: {action_name}_{file_idx}.npy")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()