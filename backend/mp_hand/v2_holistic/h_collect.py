import cv2
import mediapipe as mp
import numpy as np
import os
import time

# 1. 설정 및 초기화
actions = ['나', '너', '감사합니다', '만나다', '반갑다', '안녕하세요', '사랑합니다']
no_sequences = 15  # 단어당 촬영 횟수 (15회)
sequence_length = 30  # 한 동작당 촬영할 프레임 수 (약 1초~1.5초)
data_path = os.path.join('data') # v2_holistic/data 폴더에 저장됨

# MediaPipe Holistic 모델 설정
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 얼굴 핵심 점 인덱스 (입술 주변 + 눈썹) - 다이어트 버전
FACE_LANDMARKS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, # 입술
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300 # 눈썹
]

# 2. 좌표 추출 함수 (핵심!)
def extract_keypoints(results):
    # Pose (어깨, 팔꿈치, 손목만 추출 - 상체 6개 점)
    if results.pose_landmarks:
        # 11, 12(어깨), 13, 14(팔꿈치), 15, 16(손목)
        pose = np.array([[results.pose_landmarks.landmark[i].x, 
                          results.pose_landmarks.landmark[i].y, 
                          results.pose_landmarks.landmark[i].z] for i in [11,12,13,14,15,16]]).flatten()
    else:
        pose = np.zeros(6*3)

    # Face (미리 정한 핵심 30개 점)
    if results.face_landmarks:
        face = np.array([[results.face_landmarks.landmark[i].x, 
                          results.face_landmarks.landmark[i].y, 
                          results.face_landmarks.landmark[i].z] for i in FACE_LANDMARKS]).flatten()
    else:
        face = np.zeros(len(FACE_LANDMARKS)*3)

    # Left Hand (21개 점)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    
    # Right Hand (21개 점)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    return np.concatenate([pose, face, lh, rh])

# 3. 폴더 생성
for action in actions:
    os.makedirs(os.path.join(data_path, action), exist_ok=True)

# 4. 데이터 수집 시작
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            # --- 자동화를 위한 카운트다운 로직 ---
            for countdown in range(3, 0, -1):
                ret, frame = cap.read()
                cv2.putText(frame, f'WAIT: {countdown}s | NEXT: {action} ({sequence+1}/{no_sequences})', 
                            (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                cv2.waitKey(1000) # 1초 대기

            frames = []
            for frame_num in range(sequence_length):
                ret, frame = cap.read()

                # Holistic 추론
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 화면에 뼈대 그리기 (디버깅용)
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # 화면 상단 정보 표시
                cv2.putText(image, f'RECORDING: {action} - Frame {frame_num}', 
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)

                # 좌표 추출 및 저장
                keypoints = extract_keypoints(results)
                frames.append(keypoints)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 한 세션(30프레임) 저장
            res_path = os.path.join(data_path, action, str(sequence))
            np.save(res_path, frames)

    cap.release()
    cv2.destroyAllWindows()