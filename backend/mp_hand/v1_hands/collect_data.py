import cv2
import mediapipe as mp
import csv
import os

# 1. 초기 설정
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# 2. 저장할 파일 설정
file_name = 'hand_dataset.csv'
# 만약 파일이 없으면 헤더(제목줄)를 먼저 만듭니다.
if not os.path.exists(file_name):
    with open(file_name, mode='w', newline='') as f:
        writer = csv.writer(f)
        # 라벨(정답) + 21개 점의 x, y, z 좌표 (총 64개 열)
        header = ['label']
        for i in range(21):
            header.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(header)

# 3. 카메라 실행
cap = cv2.VideoCapture(0)

print("=== 데이터 수집 프로그램 ===")
print("1. 수어 동작을 취합니다.")
print("2. 'S' 키를 누르면 해당 좌표가 CSV에 저장됩니다.")
print("3. 종료하려면 'ESC'를 누르세요.")

# 수집할 동작의 이름을 미리 정합니다. (나중에 실행 중에 바꿀 수도 있습니다)
current_label = input("수집할 동작의 이름을 입력하세요 (예: hello, thanks): ")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 키 입력 확인
            key = cv2.waitKey(1)
            
            # 'S' 키를 누르면 저장 (대문자 S, 소문자 s 둘 다 작동하게)
            if key == ord('s') or key == ord('S'):
                # 21개 점의 좌표를 하나의 리스트로 변환
                data = [current_label]
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])
                
                # CSV 파일에 추가
                with open(file_name, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(data)
                print(f"데이터 저장 완료! (라벨: {current_label})")

    cv2.imshow("Data Collection", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()