import cv2
import mediapipe as mp
import csv
import os

# 1. 경로 및 설정
# r''를 붙여야 윈도우 경로의 백슬래시(\) 인식이 잘 됩니다.
video_dir = r'C:\Users\akfnx\Downloads\New_sample\원천데이터\REAL\WORD\15'
output_csv = 'yes_no_dataset.csv'

# 추출할 타겟 단어와 저장할 라벨 이름
target_map = {
    "WORD2318": "correct", # 맞다
    "WORD1359": "wrong"    # 틀리다
}

# 2. MediaPipe 초기화
mp_hands = mp.solutions.hands
# 학습 데이터 추출 시에는 static_image_mode를 True로 하는 것이 더 정확합니다.
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)

# 3. CSV 파일 초기화 (헤더 생성)
# utf-8-sig는 엑셀에서 한글이 깨지지 않게 해줍니다.
with open(output_csv, mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    header = ['label']
    for i in range(21):
        header.extend([f'x{i}', f'y{i}', f'z{i}'])
    writer.writerow(header)

# 4. 폴더 내 파일 필터링 (타겟 단어만 골라내기)
all_files = os.listdir(video_dir)
target_files = [f for f in all_files if any(word_id in f for word_id in target_map.keys()) and f.endswith('.mp4')]

print(f"총 {len(target_files)}개의 타겟 영상을 찾았습니다. 작업을 시작합니다.")

# 5. 영상 처리 루프
for video_name in target_files:
    # 파일명에서 단어 ID 추출 (예: NIA_SL_WORD2318_... -> WORD2318)
    video_id = ""
    for k in target_map.keys():
        if k in video_name:
            video_id = k
            break
    
    label = target_map[video_id]
    video_path = os.path.join(video_dir, video_name)
    cap = cv2.VideoCapture(video_path)
    
    print(f"[{video_name}] 처리 중... 라벨: {label}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 영상의 모든 프레임을 다 쓰면 중복이 심하므로 5프레임마다 1개씩 추출
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 5 != 0:
            continue

        # MediaPipe 처리를 위한 변환
        frame = cv2.flip(frame, 1) # 좌우 반전
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # [데이터 전처리: 정규화]
                # 손목(0번 점)의 좌표를 가져와서 모든 점의 기준점(원점)으로 삼음
                wrist = hand_landmarks.landmark[0]
                base_x, base_y, base_z = wrist.x, wrist.y, wrist.z

                row = [label]
                for lm in hand_landmarks.landmark:
                    # 모든 좌표에서 손목 좌표를 빼서 저장 (손목은 항상 0,0,0이 됨)
                    # 이렇게 하면 손이 화면 어디에 있든 상관없이 '모양'만 학습하게 됨
                    row.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                
                # CSV 파일에 즉시 추가
                with open(output_csv, mode='a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

    cap.release()

print(f"\n✅ 모든 작업 완료! '{output_csv}' 파일이 생성되었습니다.")