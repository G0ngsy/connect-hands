import cv2, mediapipe as mp, numpy as np
from tensorflow.keras.models import load_model

# 1. 모델 로드
actions = ['안녕하세요','ILOVEU','반갑다','HELLO', '좋다','BAD','만나다','IDLE']
model = load_model('suyeo_model.h5')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
seq = []

print("🚀 실시간 인식 시작! (꺼짐 방지 로직 적용)")

while cap.isOpened():
    ret, img = cap.read()
    if not ret: break
    img = cv2.flip(img, 1)
    res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 💡 항상 126개의 공간을 확보합니다.
    row = []
    if res.multi_hand_landmarks:
        for i in range(2):
            if i < len(res.multi_hand_landmarks):
                h = res.multi_hand_landmarks[i]
                base_x, base_y, base_z = h.landmark[0].x, h.landmark[0].y, h.landmark[0].z
                for lm in h.landmark:
                    row.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                mp.solutions.drawing_utils.draw_landmarks(img, h, mp_hands.HAND_CONNECTIONS)
            else:
                # 손이 하나만 보이면 나머지 63개는 0으로 채움
                row.extend([0.0] * 63)
    else:
        # 손이 아예 안 보이면 126개 모두 0으로 채움 (혹은 seq 초기화)
        row.extend([0.0] * 126)

    seq.append(row)
    if len(seq) > 62:
        seq.pop(0) # 62개 유지

    # 💡 데이터가 정확히 62개이고, 모두 0인 상태가 아닐 때만 예측
    if len(seq) == 62:
        input_data = np.array(seq, dtype=np.float32)
        
        # 💡 모델이 기대하는 (1, 62, 126) 모양인지 최종 확인
        if input_data.shape == (62, 126):
            input_data = np.expand_dims(input_data, axis=0)
            try:
                y_pred = model.predict(input_data, verbose=0).squeeze()
                i_pred = int(np.argmax(y_pred))
                if y_pred[i_pred] > 0.9:
                    cv2.putText(img, f'{actions[i_pred]} {int(y_pred[i_pred]*100)}%', (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                print(f"예측 에러 발생: {e}") # 에러 메시지 출력 후 꺼짐 방지

    cv2.imshow('Sign Language AI', img)
    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()