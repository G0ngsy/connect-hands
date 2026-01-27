import sys
import cv2

print("1. OpenCV 로딩 중...")
import mediapipe as mp
print("2. MediaPipe 로딩 완료")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

print("3. 카메라 초기화 중...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("에러: 카메라를 열 수 없습니다.")
    sys.exit()

# MediaPipe Hands 설정
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

print("4. 준비 완료! ESC 누르면 종료됩니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # 🔥 Day 2 핵심: 손 랜드마크 좌표 출력
            for idx, lm in enumerate(hand_landmarks.landmark):
                print(f"{idx}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")

            print("-------- 프레임 --------")
            break  # 손 하나만 확인

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Hand Landmark Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
