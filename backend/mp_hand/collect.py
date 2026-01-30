import cv2
import mediapipe as mp
import numpy as np
import os

# [설정] 저장 경로 및 단어 설정
output_dir = r"C:\Users\akfnx\Desktop\suhwa\results"
os.makedirs(output_dir, exist_ok=True)

# 💡 테스트하고 싶은 단어를 입력하세요 (한 번에 하나씩 촬영)
action_name = "THANKS"  # 또는 "HELLO"로 변경
max_len = 62          # 모델 규격 프레임 수

# MediaPipe 설정
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

print(f"🚀 [{action_name}] 촬영 준비 중...")
print("1. 카메라 화면을 보고 위치를 잡으세요.")
print("2. 's' 키를 누르면 62프레임 동안 녹화 및 데이터 추출이 시작됩니다.")
print("3. 'q' 키를 누르면 종료합니다.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    # 화면에 현재 가이드 표시
    cv2.putText(frame, f"Ready: {action_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Recording...', frame)

    key = cv2.waitKey(1)
    if key == ord('s'):  # 's' 키를 누르면 촬영 시작
        print(f"🔴 [{action_name}] 녹화 시작! 수어 동작을 해주세요...")
        video_landmarks = []
        
        while len(video_landmarks) < max_len:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            # MediaPipe 처리
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            row = []
            # 💡 양손(126개 좌표) 추출 로직 적용
            if res.multi_hand_landmarks:
                for i in range(2):
                    if i < len(res.multi_hand_landmarks):
                        h = res.multi_hand_landmarks[i]
                        mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)
                        base_x, base_y, base_z = h.landmark[0].x, h.landmark[0].y, h.landmark[0].z
                        for lm in h.landmark:
                            row.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                    else:
                        row.extend([0.0] * 63)
            else:
                row.extend([0.0] * 126)

            video_landmarks.append(row)
            
            # 진행 상황 표시
            cv2.putText(frame, f"Recording... {len(video_landmarks)}/{max_len}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Recording...', frame)
            cv2.waitKey(1)

        # 62프레임이 다 모이면 저장
        save_path = os.path.join(output_dir, f"{action_name}_{np.random.randint(1000)}.npy")
        np.save(save_path, np.array(video_landmarks))
        print(f"✅ 저장 완료: {save_path}")
        print("다시 촬영하려면 's', 종료하려면 'q'를 누르세요.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()