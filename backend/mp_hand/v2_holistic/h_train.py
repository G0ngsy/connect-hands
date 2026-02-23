import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# 1. 설정 및 경로
data_path = os.path.join('data') 
# 현재 data 폴더에 있는 단어 목록을 자동으로 가져옵니다.
actions = np.array([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
sequence_length = 30 # h_collect에서 설정한 프레임 수

# 라벨 맵핑 (나:0, 너:1 ...)
label_map = {label:num for num, label in enumerate(actions)}

# 2. 데이터 불러오기 및 전처리
sequences, labels = [], []


for action in actions:
    # 각 단어 폴더 안의 .npy 파일 개수 확인
    action_folder = os.path.join(data_path, action)
    files = [f for f in os.listdir(action_folder) if f.endswith('.npy')]
    
    for file in files:
        res = np.load(os.path.join(action_folder, file))
        sequences.append(res)
        labels.append(label_map[action])

# 넘파이 배열로 변환
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# 데이터 분리 (학습용 95%, 테스트용 5%)
# 단어 1개일 때는 분리가 무의미하지만, 코드가 멈추지 않게 설정
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

print(f"데이터 모양: {X.shape}") # (전체데이터수, 30, 315) 형태여야 함
print(f"인식된 단어: {actions}")

# 3. 모델 구축 (LSTM)
model = Sequential()
# Input_shape의 315는 (Pose 99 + Face 90 + LH 63 + RH 63)의 합계입니다.
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 315)))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax')) # 단어 개수만큼 출력

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 4. 학습 시작
print("🚀 학습을 시작합니다...")
model.fit(X_train, y_train, epochs=200, callbacks=[TensorBoard(log_dir='./logs')])

# 5. 모델 저장
os.makedirs('models', exist_ok=True)
model.save(os.path.join('models', 'holistic_model.h5'))
print("✅ 모델 저장 완료: models/holistic_model.h5")