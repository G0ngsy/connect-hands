import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ==========================================
# 1. 경로 설정 및 실제 데이터 폴더 필터링
# ==========================================
data_path = os.path.join('data') 

# [중요] 비어있는 폴더는 제외하고, 실제 .npy 파일이 들어있는 폴더만 가져옵니다.
actions = []
for d in os.listdir(data_path):
    folder_path = os.path.join(data_path, d)
    if os.path.isdir(folder_path):
        # 폴더 안에 .npy 파일이 하나라도 있는지 확인
        files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        if len(files) > 0:
            actions.append(d)

actions = np.array(sorted(actions)) # 가나다순 정렬

print(f"✅ 학습을 진행할 실제 단어 리스트 ({len(actions)}개): {actions}")

label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []

# ==========================================
# 2. 데이터 불러오기
# ==========================================
for action in actions:
    folder_path = os.path.join(data_path, action)
    files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    for file in files:
        res = np.load(os.path.join(folder_path, file))
        sequences.append(res)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# 데이터가 부족할 경우를 대비해 분리 비율 조정
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# ==========================================
# 3. LSTM 모델 구축 (Input: 177)
# ==========================================
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 177)))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
# 출력 노드 수를 현재 데이터가 있는 단어 개수(len(actions))로 맞춥니다.
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# ==========================================
# 4. 학습 시작
# ==========================================
print(f"🚀 {len(actions)}개 단어 학습을 시작합니다!")
model.fit(X_train, y_train, epochs=200)

# 모델 저장 폴더 생성 및 저장
os.makedirs('models', exist_ok=True)
model.save(os.path.join('models', 'holistic_model.h5'))
print("✅ 모델 저장 완료: models/holistic_model.h5")