import numpy as np
import os

data_path = 'data'
actions = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

for action in actions:
    files = os.listdir(os.path.join(data_path, action))
    for file in files:
        if file.endswith('.npy'):
            path = os.path.join(data_path, action, file)
            data = np.load(path)
            if data.shape != (30, 177):
                print(f"❌ 발견! 모양이 다른 파일: {path} (현재 모양: {data.shape})")