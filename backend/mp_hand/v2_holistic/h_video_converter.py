import cv2
import mediapipe as mp
import numpy as np
import os

# [설정]
RAW_PATH = 'raw_videos'  # 폰 영상이 들어있는 폴더
DATA_PATH = 'data'        # 좌표 데이터가 저장될 폴더
MAX_LEN = 30              # 한 동작당 프레임 수

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# [동일 규격] 기존에 사용한 177개 좌표 추출 함수
def extract_keypoints(results):
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

# 메인 처리 로직
actions = [d for d in os.listdir(RAW_PATH) if os.path.isdir(os.path.join(RAW_PATH, d))]

for action in actions:
    video_dir = os.path.join(RAW_PATH, action)
    output_dir = os.path.join(DATA_PATH, action)
    os.makedirs(output_dir, exist_ok=True)
    
    videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.mov', '.avi'))]
    print(f"🎬 단어 [{action}] 처리 시작... (영상 {len(videos)}개)")

    for v_idx, v_name in enumerate(videos):
        cap = cv2.VideoCapture(os.path.join(video_dir, v_name))
        video_landmarks = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 폰 영상은 회전되어 있을 수 있으므로 필요한 경우 처리
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            keypoints = extract_keypoints(results)
            video_landmarks.append(keypoints)
            
            # 30프레임이 다 모이면 중단 (영상 앞부분 1초만 사용)
            if len(video_landmarks) == MAX_LEN: break
        
        cap.release()
        
        # 30프레임이 꽉 찼을 때만 저장
        if len(video_landmarks) == MAX_LEN:
            file_num = len([f for f in os.listdir(output_dir) if f.endswith('.npy')])
            np.save(os.path.join(output_dir, f"{file_num}.npy"), np.array(video_landmarks))
            print(f"  > {v_name} 변환 완료 ({file_num}.npy)")
        else:
            print(f"  > ⚠️ {v_name} 실패: 프레임 부족 ({len(video_landmarks)}/{MAX_LEN})")

print("\n✨ 모든 영상 데이터 변환이 완료되었습니다!")