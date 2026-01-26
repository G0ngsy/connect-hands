import cv2
import mediapipe as mp

# 1. MediaPipe 관련 초기 설정
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# 2. Hands 모델 설정
# - static_image_mode: 비디오 스트림이므로 False
# - max_num_hands: 인식할 최대 손 개수
# - min_detection_confidence: 인식 신뢰도 임계값
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 3. 카메라 실행 (기본 카메라 0번)
cap = cv2.VideoCapture(0)

print("프로그램이 시작되었습니다. 종료하려면 웹캠 창에서 ESC 키를 누르세요.")

while True:
    # 카메라 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("카메라 프레임을 읽을 수 없습니다.")
        break

    # 화면 좌우 반전 (거울 모드)
    frame = cv2.flip(frame, 1)
    
    # MediaPipe는 RGB 이미지를 사용하므로 BGR -> RGB 변환
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 손 인식 처리
    result = hands.process(rgb)

    # 손이 인식되었을 때 랜드마크 그리기
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )

    # 결과 화면 출력
    cv2.imshow("Hand Tracking Test", frame)
    
    # ESC 키(27)를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 4. 자원 해제 및 종료
cap.release()
cv2.destroyAllWindows()
print("프로그램이 정상적으로 종료되었습니다.")