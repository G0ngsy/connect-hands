import cv2
import mediapipe as mp
import numpy as np
import os
import json

# [설정] 모든 파일이 들어있는 경로로 수정
base_dir = r"C:\Users\akfnx\Desktop\suhwa" 
output_dir = r"C:\Users\akfnx\Desktop\suhwa\results"
os.makedirs(output_dir, exist_ok=True)

# MediaPipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

print("🚀 [데이터 추출] 단일 폴더 모드 시작...")

# 1. 폴더 내의 모든 파일 목록 가져오기
files = os.listdir(base_dir)
# 그 중 json 파일만 필터링
json_files = [f for f in files if f.endswith("_morpheme.json")]

if not json_files:
    print(f"❌ '{base_dir}' 경로에서 _morpheme.json 파일을 찾지 못했습니다.")
    print(f"현재 폴더 내 파일 예시: {files[:5]}") # 디버깅용 출력

for json_name in json_files:
    # 2. 짝이 맞는 mp4 파일 이름 만들기
    video_name = json_name.replace("_morpheme.json", ".mp4")
    video_path = os.path.join(base_dir, video_name)
    json_path = os.path.join(base_dir, json_name)

    # 영상 파일이 실제로 있는지 확인
    if not os.path.exists(video_path):
        print(f"⚠️ 영상 없음(건너뜀): {video_name}")
        continue

    print(f"📦 처리 중: {video_name}")

    # 3. JSON에서 구간 정보 읽기
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            start_t = content['data'][0]['start']
            end_t = content['data'][0]['end']
    except Exception as e:
        print(f"   ❌ JSON 읽기 에러: {e}")
        continue

    # 4. 영상 처리 및 좌표 추출
    cap = cv2.VideoCapture(video_path)
    video_landmarks = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        curr_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # 지정된 수어 구간 프레임만 추출
        if start_t <= curr_sec <= end_t:
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            row = []
            if res.multi_hand_landmarks:
                for i in range(2):
                    if i < len(res.multi_hand_landmarks):
                        h = res.multi_hand_landmarks[i]
                        base_x, base_y, base_z = h.landmark[0].x, h.landmark[0].y, h.landmark[0].z
                        for lm in h.landmark:
                            row.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                    else:
                        row.extend([0.0] * 63)
            else:
                row.extend([0.0] * 126)
            
            video_landmarks.append(row)
    
    cap.release()

    # 5. 결과 저장 (파일명에서 확장자 제외하고 저장)
    if video_landmarks:
        save_name = json_name.replace("_morpheme.json", ".npy")
        np.save(os.path.join(output_dir, save_name), np.array(video_landmarks))
        print(f"   ✅ 저장 성공: {save_name} (프레임: {len(video_landmarks)})")

print("\n✨ 모든 작업이 완료되었습니다!")