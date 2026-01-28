import cv2, mediapipe as mp, numpy as np, os, json

base_dir = r"C:\Users\akfnx\Desktop\suhwa" 
output_dir = r"C:\Users\akfnx\Desktop\suhwa\results"
os.makedirs(output_dir, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

print("🚀 [1단계] 양손 상대 좌표 데이터 추출 시작...")

for root, dirs, files in os.walk(base_dir):
    for filename in files:
        if filename.endswith("_morpheme.json") and "WORD15" in filename:
            video_path = os.path.join(root, filename.replace("_morpheme.json", ".mp4"))
            if not os.path.exists(video_path): continue

            with open(os.path.join(root, filename), 'r', encoding='utf-8') as f:
                content = json.load(f)
                start_t, end_t = content['data'][0]['start'], content['data'][0]['end']

            cap = cv2.VideoCapture(video_path)
            video_landmarks = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                curr_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if start_t <= curr_sec <= end_t:
                    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    
                    row = []
                    # 💡 양손(최대 2개) 처리 로직
                    for i in range(2):
                        if res.multi_hand_landmarks and i < len(res.multi_hand_landmarks):
                            h = res.multi_hand_landmarks[i]
                            base_x, base_y, base_z = h.landmark[0].x, h.landmark[0].y, h.landmark[0].z
                            for lm in h.landmark:
                                row.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                        else:
                            row.extend([0.0] * 63) # 손이 없으면 0으로 채움
                    
                    if any(v != 0.0 for v in row): # 최소 한 손이라도 감지된 경우만 저장
                        video_landmarks.append(row)
            cap.release()
            if video_landmarks:
                np.save(os.path.join(output_dir, f"{filename.replace('.json', '')}.npy"), np.array(video_landmarks))
                print(f"✅ 양손 저장 완료: {filename}")