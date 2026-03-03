import { useState, useEffect } from 'react';
import Header from './components/Header';
import StartScreen from './components/StartScreen';
import HistoryLog from './components/HistoryLog';
import CameraView from './components/CameraView';

// ==========================================
// 1. 데이터 타입 정의 (Interface)
// ==========================================
/**
 * 백엔드(Python) 서버에서 웹소켓으로 보내주는 데이터의 규격입니다.
 * 파이썬의 result_json 구조와 리액트의 인터페이스가 정확히 일치해야 에러가 나지 않습니다.
 */
export interface SignResult {
  word: string;        // 현재 인식 중인 단어 (예: "나", "너")
  confidence: number;  // 인식 신뢰도 (0~1 사이 소수점)
  sentence: string;    // 지금까지 인식된 단어들의 나열 (Gloss Sequence)
  translated: string[]; // AI(EXAONE)가 의역한 3가지 문장 후보 리스트
  image: string;       // 백엔드에서 보낸 실시간 카메라 이미지 (Base64 텍스트)
  timestamp?: string;  // 로그 저장을 위한 시간 정보 (선택 사항)
}

function App() {
  // ==========================================
  // 2. 상태 관리 (State Management)
  // ==========================================
  const [isStarted, setIsStarted] = useState(false);       // 카메라와 인식 서비스를 시작했는지 여부
  const [currentResult, setCurrentResult] = useState<SignResult | null>(null); // 백엔드에서 받은 최신 데이터 저장
  const [logs, setLogs] = useState<SignResult[]>([]);     // 사용자가 선택하여 저장된 문장 기록들
  const [isLoading, setIsLoading] = useState(true);       // 앱 초기 실행 시 로딩 애니메이션 표시 여부

  // ==========================================
  // 3. 앱 초기화 및 스플래시 화면 제어
  // ==========================================
  useEffect(() => {
    // 앱이 켜지면 2초 동안 'CONNECTING HANDS...' 로딩 화면을 보여줍니다.
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 2000); 

    return () => clearTimeout(timer); // 메모리 누수 방지
  }, []);

  // ==========================================
  // 4. 실시간 웹소켓 통신 (Backend 연동)
  // ==========================================
  useEffect(() => {
    let socket: WebSocket | null = null;

    // '시작하기' 버튼을 눌렀을 때만 백엔드 서버와 연결을 시도합니다.
    if (isStarted) {
      // 파이썬 서버 주소 (h_main.py에서 설정한 8080 포트)
      socket = new WebSocket('ws://127.0.0.1:8080/ws');

      // 서버로부터 실시간 데이터를 받았을 때 실행되는 함수
      socket.onmessage = (event) => {
        const data: SignResult = JSON.parse(event.data);
        setCurrentResult(data); // 받은 데이터를 상태에 저장하여 화면에 반영
      };

      // 연결 중 에러 발생 시 로그 출력
      socket.onerror = (err) => console.error("❌ 서버 연결 에러:", err);
    }

    // 컴포넌트가 꺼지거나 '홈'으로 돌아갈 때 웹소켓 연결을 안전하게 닫습니다.
    return () => {
      if (socket) socket.close();
    };
  }, [isStarted]); // isStarted 값이 바뀔 때마다 실행

  // ==========================================
  // 5. 비즈니스 로직 (문장 선택 및 기록)
  // ==========================================
  /**
   * AI가 추천한 3가지 문장 중 하나를 클릭했을 때 실행되는 함수입니다.
   * 선택한 문장을 하단 HistoryLog에 추가합니다.
   */
  const handleSelectLog = (selectedText: string) => {
    if (!currentResult) return;

    // 새로운 로그 데이터 생성
    const newLog: SignResult = {
      ...currentResult,
      translated: [selectedText], // 여러 후보 중 사용자가 '선택한 딱 하나의 문장'만 저장
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setLogs((prev) => {
      // 연속으로 똑같은 문장을 클릭하는 것 방지 (중복 제거)
      if (prev.length > 0 && prev[0].translated[0] === selectedText) return prev;
      // 최신순으로 10개까지만 기록 유지
      return [newLog, ...prev].slice(0, 10);
    });
  };

  // ==========================================
  // 6. UI 렌더링 (UI Rendering)
  // ==========================================

  // [상태 1] 초기 로딩(스플래시) 화면
  if (isLoading) {
    return (
      <div className="fixed inset-0 bg-[#FFF5F8] flex flex-col items-center justify-center z-[10000]">
        <div className="relative">
          {/* 로고 애니메이션 */}
          <div className="w-20 h-20 bg-gradient-to-br from-[#FF66B2] to-[#4AD799] 
            rounded-3xl flex items-center justify-center shadow-2xl animate-bounce">
            <span className="text-white font-black text-2xl tracking-tighter">CH</span>
          </div>
          <div className="absolute inset-0 bg-pink-200 rounded-full animate-ping opacity-20"></div>
        </div>
        <p className="mt-8 text-[#FF66B2] font-black tracking-[0.3em] animate-pulse">
          CONNECTING HANDS...
        </p>
      </div>
    );
  }

  // [상태 2] 실제 앱 메인 화면
  return (
    <div className="min-h-screen bg-[#FFF5F8] font-sans text-[#330019]">
      {/* 상단 헤더: 로고 클릭 시 setIsStarted(false)를 실행해 홈 화면으로 보냅니다. */}
      <Header onGoHome={() => {
          setIsStarted(false);     // 홈으로 이동
          setLogs([]);             // [핵심] 로그 기록 초기화  
          setCurrentResult(null);  // 현재 인식 결과 비우기
          console.log("🧹 모든 기록이 리셋되었습니다.");
      }} />

      <main className="max-w-4xl mx-auto pt-2 pb-12 space-y-8 px-4">
        {!isStarted ? (
          /* [홈 모드] 시작 버튼이 있는 대기 화면 */
          <StartScreen onStart={() => setIsStarted(true)} />
        ) : (
          /* [인식 모드] 카메라 및 AI 결과 화면 */
          <div className="space-y-8 animate-in fade-in duration-1000">
            {/* 실시간 카메라 뷰: 백엔드에서 보낸 이미지를 표시 */}
            <CameraView serverImage={currentResult?.image} />
            
            <div className="text-center space-y-6">
              {/* 1. 현재 인식 중인 단어 정보 */}
              <div>
                <p className="text-[#FF66B2] font-bold text-xs uppercase mb-2 tracking-widest">Current Word</p>
                <h2 className="text-6xl font-black italic transition-all duration-300">
                  {currentResult?.word || "..."}
                </h2>
                {currentResult && currentResult.confidence > 0 && (
                  <span className="inline-block mt-4 px-4 py-1 bg-white border border-pink-100 rounded-full text-pink-500 font-bold text-sm shadow-sm">
                    {(currentResult.confidence * 100).toFixed(1)}% Match
                  </span>
                )}
              </div>

              {/* 2. 번역 결과 선택 카드 */}
              <div className="max-w-2xl mx-auto p-8 bg-white/80 backdrop-blur-sm rounded-[40px] shadow-xl border border-white">
                {/* 단어 나열 (Gloss) 표시 구역 */}
                <div className="mb-6 text-left border-b border-pink-50 pb-4">
                  <p className="text-[#330019]/40 text-[10px] font-black uppercase tracking-widest mb-2">Word Sequence</p>
                  <p className="text-xl font-bold text-[#330019]/70">
                    {currentResult?.sentence || "수어를 시작해 주세요."}
                  </p>
                </div>

                {/* AI 문장 선택 버튼 구역 */}
                <div className="pt-2 text-left">
                  <p className="text-[#4AD799] text-[10px] font-black uppercase tracking-widest mb-4">
                    AI Interpretation (Click to Save Log)
                  </p>
                  <div className="flex flex-col gap-3">
                    {currentResult?.translated && currentResult.translated.length > 0 ? (
                      // 백엔드에서 보낸 3가지 문장 배열을 순회하며 버튼 생성
                      currentResult.translated.map((option, index) => (
                        <button
                          key={index}
                          onClick={() => handleSelectLog(option)}
                          className="w-full text-left p-4 rounded-2xl bg-[#FFF5F8] hover:bg-[#FF66B2] hover:text-white 
                                     transition-all duration-200 font-bold text-lg border border-pink-100 shadow-sm
                                     active:scale-[0.95] group"
                        >
                          <span className="opacity-40 mr-3 group-hover:text-white/60">0{index + 1}</span>
                          {option}
                        </button>
                      ))
                    ) : (
                      <span className="text-gray-300 italic animate-pulse">문장을 생성하는 중입니다...</span>
                    )}
                  </div>
                </div>
              </div>

              {/* 3. 문장 기록 로그 리스트 */}
              <HistoryLog logs={logs} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;