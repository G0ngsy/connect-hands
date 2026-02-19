import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image  # 한글 출력을 위한 라이브러리

# [설정] 저장 경로 및 단어 설정
output_dir = r"C:\Users\akfnx\Desktop\suhwa\results"
os.makedirs(output_dir, exist_ok=True)

# 💡 테스트하고 싶은 단어 설정
action_name = "IDLE"  
max_len = 62          # 모델 규격 프레임 수 (데이터 일관성을 위해 고정)

# --- 한글 출력을 위한 함수 정의 ---
def draw_text(img, text, position, font_size=30, color=(255, 255, 255)):
    """
    OpenCV 이미지에 한글을 그려주는 함수
    """
    # OpenCV(BGR) 이미지를 PIL(RGB) 이미지로 변환
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 윈도우용 맑은고딕 폰트 경로 설정 (폰트 파일이 없을 경우 기본폰트 사용)
    try:
        font = ImageFont.truetype("malgun.ttf", font_size)
    except:
        font = ImageFont.load_default()
        
    # 텍스트 그리기
    draw.text(position, text, font=font, fill=color)
    
    # 다시 OpenCV(BGR) 이미지로 변환하여 반환
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# MediaPipe 설정
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

print(f"🚀 [{action_name}] 촬영 준비 중...")
print("1. 카메라 화면을 보고 위치를 잡으세요.")
print("2. 's' 키를 누르면 62프레임 동안 녹화 및 데이터 추출이 시작됩니다.")
print("3. 'q' 키를 누르면 종료합니다.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1) # 좌우 반전
    
    # 화면 상단에 한글 가이드 표시
    frame = draw_text(frame, f"대기 중: {action_name}", (10, 30), font_size=35, color=(255, 255, 0))
    frame = draw_text(frame, "촬영하려면 'S'를 누르세요", (10, 80), font_size=20)
    
    cv2.imshow('Hand Data Collector', frame)

    key = cv2.waitKey(1)
    
    # --- 's' 키를 누르면 데이터 수집 시작 ---
    if key == ord('s'):
        print(f"🔴 [{action_name}] 녹화 시작! 수어 동작을 해주세요...")
        video_landmarks = []
        
        while len(video_landmarks) < max_len:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            
            # MediaPipe 처리 (BGR -> RGB 변환 필요)
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            row = []
            # 양손 좌표 추출 (한 손당 21개 점 * xyz 3축 = 63개 / 양손 총 126개)
            if res.multi_hand_landmarks:
                # 감지된 손들의 랜드마크를 순차적으로 추출
                for i in range(2):
                    if i < len(res.multi_hand_landmarks):
                        h = res.multi_hand_landmarks[i]
                        # 화면에 뼈대 그리기
                        mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)
                        
                        # 0번 좌표(손목)를 기준으로 상대 좌표 계산 (원점 정규화)
                        base_x, base_y, base_z = h.landmark[0].x, h.landmark[0].y, h.landmark[0].z
                        for lm in h.landmark:
                            row.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                    else:
                        # 한 손만 감지된 경우 나머지는 0으로 채움
                        row.extend([0.0] * 63)
            else:
                # 손이 아예 안 보일 경우 전체를 0으로 채움
                row.extend([0.0] * 126)

            video_landmarks.append(row)
            
            # 녹화 진행률 화면 표시
            frame = draw_text(frame, f"녹화 중... {len(video_landmarks)}/{max_len}", (10, 30), color=(0, 0, 255))
            cv2.imshow('Hand Data Collector', frame)
            cv2.waitKey(1)

        # --- 62프레임 수집 완료 후 저장 ---
        save_path = os.path.join(output_dir, f"{action_name}_{np.random.randint(10000)}.npy")
        np.save(save_path, np.array(video_landmarks))
        print(f"✅ 저장 완료: {save_path}")
        print("다시 촬영하려면 's', 종료하려면 'q'를 누르세요.")

    # --- 'q' 키를 누르면 종료 ---
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()