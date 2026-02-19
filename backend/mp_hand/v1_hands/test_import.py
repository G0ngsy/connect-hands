import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
print("1. 환경 변수 설정 완료")

try:
    import cv2
    print("2. OpenCV 로드 성공")
    import mediapipe as mp
    print("3. MediaPipe 로드 성공")
    import tensorflow as tf
    print("4. TensorFlow 로드 성공")
    import google.generativeai as genai
    print("5. Gemini 로드 성공")
except Exception as e:
    print(f"❌ 에러 발생 지점: {e}")