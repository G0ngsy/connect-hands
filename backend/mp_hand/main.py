import os
import sys

print("1. 모듈 로딩 시작...")
import cv2
print("2. OpenCV 로드 완료")

# mediapipe 전체 대신 필요한 모듈만 직접 임포트 시도
try:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_draw
    print("3. MediaPipe 손 인식 모듈 로드 완료!")
except Exception as e:
    print(f"에러 발생: {e}")
    sys.exit()

print("4. 카메라 초기화 중...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("에러: 카메라를 열 수 없습니다.")
    sys.exit()

# 모델 설정
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

print("5. 모든 준비 완료! 웹캠 창이 뜹니다.")

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 인식 실행
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Success!", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()