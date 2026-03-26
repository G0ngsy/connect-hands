# 🤟 CONNECTING HANDS
> **실시간 MediaPipe Holistic 관절 추출 및 로컬 LLM 기반 지능형 수어 통역 서비스**

---

## 1. 프로젝트 개요 (Overview) 📢
**Connect Hands**는 청각장애인과 비장애인 사이의 소통 장벽을 낮추기 위한 **실시간 수어 통역·보조 서비스**입니다.
* **핵심 기능**: 카메라 기반 수어 인식과 음성 인식(STT)을 결합
* **기대 효과**: 수어 ↔ 음성/텍스트를 즉각적으로 변환하여 일상 및 공공기관 환경에서의 원활한 대화 지원

---

## 2. 사용자 경험 흐름 (User Flow) 🔄
사용자가 서비스를 직관적으로 이용할 수 있도록 설계된 5단계 프로세스입니다.

1. **진입** 🏁: 스플래시 화면 후 메인 홈에서 `[시작하기]` 클릭
2. **인식** 📸: 카메라 화면에 실시간 **뼈대(Skeleton) 피드백** 노출 및 수어 동작 수행
3. **생성** ✨: 하단에 EXAONE 3.5가 생성한 **3가지 문장 스타일** 버튼 실시간 노출
4. **확정** ✅: 원하는 스타일의 문장을 클릭하여 `History Log`에 정식 기록
5. **초기화** ♻️: 손을 내리고 일정 시간 대기 시 데이터 시퀀스 자동 리셋

### 📸화면 미리보기
<div align="center">
    <img src="frontend\src\assets\ConnectHands.gif" width="200" />
</div>
---

## 3. 핵심 기능 (Core Features) 🚀
* **실시간 전신 관절 추적**: `MediaPipe Holistic`을 이용한 얼굴/몸/손의 정밀 트래킹.
* **지능형 수어 단어 인식**: `LSTM` 기반 시퀀스 학습 모델을 통한 동적 수어 인식.
* **로컬 LLM 문장 의역**: **EXAONE 3.5**를 활용해 단어 나열을 자연스러운 문장으로 변환.
* **맞춤형 3대 문체 제공**: 상황에 맞는 **격식/친근/간결** 스타일 선택 기능.
* **오작동 방지 안전장치**: 배꼽(골반) 라인 기준 인식 제어 로직 탑재로 오인식 최소. 

---

## 4. 시스템 아키텍처 (Architecture) 🏗️

<p align="center">
<img width="1122" height="493" alt="수어-페이지-1 drawio" src="https://github.com/user-attachments/assets/0db5b0b6-269f-4363-a60f-ef5d10ce93bb" />
</p>

### 🔄 전체 데이터 파이프라인 (End-to-End Flow)
1.  **Input**: 사용자의 수어 동작을 카메라로 캡처 (React Frontend).
2.  **Streaming**: `WebSocket`을 통해 실시간 프레임 데이터를 백엔드로 전송.
3.  **Extraction**: `MediaPipe Holistic`을 통해 핵심 관절(177개) 좌표 추출.
4.  **Recognition**: 누적된 시퀀스 데이터를 `TensorFlow LSTM` 모델로 단어 분류.
5.  **Generation**: 인식된 단어들을 `Ollama(EXAONE 3.5)`에 전달하여 3가지 스타일 문장 생성.
6.  **Output**: 최종 생성된 문장 데이터를 프론트엔드로 송신 및 사용자 선택. 


### 🛠️ 모듈별 상세 스택 및 역할

| Layer | Component | Key Technologies | Description |
| :--- | :--- | :--- | :--- |
| **Client** | Frontend | React, TypeScript, Tailwind | 실시간 영상 렌더링 및 UI 인터랙션 |
| **Server** | Backend | FastAPI, WebSocket | 비동기 데이터 스트리밍 및 파이프라인 관리 |
| **Vision** | Preprocessing | MediaPipe Holistic | 177개 랜드마크 추출 및 데이터 정규화 |
| **AI (DL)** | Action Recognition | TensorFlow (LSTM) | 시계열 동작 인식을 통한 수어 단어 분류 |
| **AI (LLM)** | Sentence Generation | EXAONE 3.5 (Ollama) | 로컬 환경에서의 지능형 문장 교정 및 의역 |


### 💡 아키텍처 설계 포인트 (Reverse Engineering)
* **비동기 처리**: `asyncio`를 활용하여 LLM이 문장을 생성하는 동안에도 카메라 프레임 수신이 중단되지 않도록 설계되었습니다.
* **데이터 다이어트**: 전체 468개 좌표 대신 수어에 필수적인 177개 좌표만 사용하여 시스템 부하를 대폭 낮췄습니다.
* **보안성**: 모든 추론 과정을 로컬(RTX 4070)에서 처리하여 대화 내용의 외부 유출을 원천 차단했습니다.

---

## 5. 핵심 기능 상세 리스트 (Feature Details) 📋

| 기능명 | 상세 내용 | 비고 |
| :--- | :--- | :--- |
| **Data Diet (177 pts)** | 468개 전체 좌표 중 수어 핵심 177개만 추출 | 연산 속도 최적화 |
| **Relative Normalization** | 손목 기준 상대 좌표계 적용 | 거리 무관 인식 구현 |
| **Hip-line Safety** | 손목이 골반 아래일 때 추론 엔진 일시 정지 | 오인식 방지 |
| **Async Translation** | `asyncio` Task 처리를 통한 논블로킹 통역 | 카메라 끊김 해결 |
| **Auto Session Reset** | 10초간 IDLE 상태 유지 시 컨텍스트 초기화 | 세션 자동 관리 |

---

## 6. UI/UX 구성 (Interface Design) 🎨
* **Live AI Tracking View**: 사용자의 움직임을 실시간 뼈대(Skeleton)로 피드백하여 신뢰감 형성.
* **Confidence Badge**: 인공지능의 확신도를 `%`로 시각화하여 데이터 투명성 제공.
* **3-Style Selection Buttons**: 사용자가 능동적으로 통역 결과에 참여하는 인터랙티브 요소.
* **History Log Card**: 정제된 통역 기록을 시간순으로 관리하는 아카이빙 UI.

---

## 7. 기술 스택 (Tech Stack) 💻
* **Frontend**: `React`, `TypeScript`, `Tailwind CSS`, `Lucide React`
* **Backend**: `FastAPI`, `WebSocket`, `Uvicorn`, `Python-dotenv`
* **AI/ML Engine**: `MediaPipe Holistic`, `TensorFlow 2.12 (Keras/LSTM)`, `NumPy`
* **Local LLM**: `Ollama`, `EXAONE 3.5 (7.8B Parameter)`

---

## 8. 기술 선정 이유 (Why This Tech) 💡

### **MediaPipe Holistic**
> Hands 모델보다 넓은 시야를 제공하면서도, 전체 Mesh 모델보다 가벼워 실시간성에 가장 적합합니다.

### **LSTM (Long Short-Term Memory)**
> 수어는 '시간적 흐름'이 중요합니다. 이전 프레임을 기억하는 LSTM이 동작 인식의 정확도를 보장합니다.

### **EXAONE 3.5:7.8B (Local LLM)**
> 로컬 환경(RTX 4070)을 활용하여 보안성과 비용 문제를 동시에 해결하고, 한국어 정서에 최적화된 의역 성능을 확보했습니다.

### **WebSocket**
> 1초에 30번 이상의 데이터를 실시간으로 주고받기 위해 HTTP보다 빠른 양방향 통로가 필수적입니다.

---

## 9.실행 방법(Usage)
```
1. Ollama 실행
    ollama run exaone3.5:7.8b

2. Backend
    cd backend/mp_hand/v2_holistic -> python h_main.py

3. Frontend
    cd frontend -> npm run dev
```
