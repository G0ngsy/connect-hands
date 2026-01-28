import numpy as np, os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

actions = ['WORD1501', 'WORD1502', 'WORD1503', 'WORD1504', 'WORD1505', 
           'WORD1506', 'WORD1507', 'WORD1508', 'WORD1509', 'WORD1510', 
           'WORD1511', 'WORD1512', 'WORD1513', 'WORD1514', 'WORD1520']
data_path = r"C:\Users\akfnx\Desktop\suhwa\results"
max_len = 62 

X, y = [], []
for i, action in enumerate(actions):
    count = 0
    for f in os.listdir(data_path):
        if action in f and f.endswith('.npy'):
            data = np.load(os.path.join(data_path, f))
            
            # 💡 양손 데이터 규격(126)에 맞게 패딩/자르기
            if len(data) > max_len: 
                data = data[:max_len]
            elif len(data) < max_len:
                # 데이터가 양손(126) 규격인지 확인하며 패딩
                pad = np.zeros((max_len - len(data), 126)) 
                data = np.concatenate([data, pad], axis=0)
            
            # 데이터 모양이 (62, 126)인 경우만 추가 (에러 방지)
            if data.shape == (62, 126):
                X.append(data)
                y.append(i)
                count += 1
    print(f"✅ {action}: {count}개 로드 완료")

X = np.array(X)
y = to_categorical(y, num_classes=len(actions))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 💡 모델 정의 (이미 126이 적용되어 있습니다)
model = Sequential([
    LSTM(128, return_sequences=True, activation='relu', input_shape=(max_len, 126)),
    Dropout(0.2),
    LSTM(256, return_sequences=False, activation='relu'),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("🚀 [2단계] 양손 모델 학습 시작...")
model.fit(X_train, y_train, epochs=300, batch_size=16, validation_data=(X_test, y_test))

# 💡 기존 모델 덮어쓰기
model.save('suyeo_model.h5')
print("✨ 학습 완료 및 suyeo_model.h5 저장 성공!")