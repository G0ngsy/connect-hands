import { useState, useEffect } from 'react';
import Header from './components/Header';
import StartScreen from './components/StartScreen';
import HistoryLog from './components/HistoryLog';
import CameraView from './components/CameraView';

// [1] 데이터 타입 정의: 백엔드에서 오는 데이터 구조와 일치시킵니다.
export interface SignResult {
  word: string;
  confidence: number;
  sentence: string;
  translated: string;
  image: string;
  is_detected?: boolean;
  timestamp?: string;
}

function App() {
  // --- 상태 관리 (State) ---
  const [isStarted, setIsStarted] = useState(false); // 카메라/인식 시작 여부
  const [currentResult, setCurrentResult] = useState<SignResult | null>(null); // 실시간 인식 데이터
  const [logs, setLogs] = useState<SignResult[]>([]); // 인식된 단어들의 기록(로그)
  const [isLoading, setIsLoading] = useState(true); // 앱 초기 실행 로딩 상태

  // --- [효과 1] 초기 스플래시 화면 제어 ---
  useEffect(() => {
    // 앱이 켜지면 2초 동안 로딩 화면을 보여주고 꺼집니다.
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 2000); 

    return () => clearTimeout(timer);
  }, []);

  // --- [효과 2] 실시간 웹소켓 통신 제어 ---
  useEffect(() => {
    let socket: WebSocket | null = null;

    if (isStarted) {
      // 카메라 켜기(시작) 버튼을 눌렀을 때만 백엔드 서버와 연결합니다.
      socket = new WebSocket('ws://127.0.0.1:8080/ws');

      socket.onmessage = (event) => {
        const data: SignResult = JSON.parse(event.data);
        setCurrentResult(data);

        // 신뢰도가 90% 이상이고 의미 있는 단어일 때만 로그(HistoryLog)에 추가
        if (data.confidence > 0.90 && data.word !== '' && data.word !== 'IDLE') {
          setLogs((prev) => {
            // 똑같은 단어가 연속으로 찍히는 것 방지
            if (prev.length > 0 && prev[0].word === data.word) return prev;
            
            const newLog = {
              ...data,
              timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            };
            return [newLog, ...prev].slice(0, 10); // 최신순 10개 유지
          });
        }
      };

      socket.onerror = (err) => console.error("연결 에러:", err);
    }

    // 컴포넌트가 꺼지거나 isStarted가 바뀌면 연결을 닫습니다.
    return () => {
      if (socket) socket.close();
    };
  }, [isStarted]);


  // --- [렌더링 1] 앱 초기 실행 시 보여줄 스플래시 화면 ---
  if (isLoading) {
    return (
      <div className="fixed inset-0 bg-[#FFF5F8] flex flex-col items-center justify-center z-[10000]">
        <div className="relative">
          {/* 로고 박스 애니메이션 */}
          <div className="w-20 h-20 bg-gradient-to-br from-[#FF66B2] to-[#4AD799] 
            rounded-3xl flex items-center justify-center shadow-2xl animate-bounce">
            <span className="text-white font-black text-2xl tracking-tighter">CH</span>
          </div>
          {/* 로고 뒤 파동 효과 */}
          <div className="absolute inset-0 bg-pink-200 rounded-full animate-ping opacity-20"></div>
        </div>
        <p className="mt-8 text-[#FF66B2] font-black tracking-[0.3em] animate-pulse">
          CONNECTING HANDS...
        </p>
      </div>
    );
  }

  // --- [렌더링 2] 실제 서비스 화면 ---
  return (
    <div className="min-h-screen bg-[#FFF5F8] font-sans text-[#330019]">
      {/* 상단 헤더: 로고 클릭 시 홈(setIsStarted false)으로 이동 */}
      <Header onGoHome={() => setIsStarted(false)} />

      <main className="max-w-4xl mx-auto pt-2 pb-12 space-y-8">
        {!isStarted ? (
          /* [홈 모드] 시작 버튼이 있는 대기 화면 */
          <StartScreen onStart={() => setIsStarted(true)} />
        ) : (
          /* [인식 모드] 실시간 카메라 및 결과 출력 화면 */
          <div className="space-y-8 animate-in fade-in duration-1000">
            {/* 실시간 카메라 뷰 */}
            <CameraView serverImage={currentResult?.image} />
            
            <div className="text-center space-y-6">
              {/* 현재 인식된 단어 표시 영역 */}
              <div>
                <p className="text-[#FF66B2] font-bold text-xs uppercase mb-2 tracking-widest">
                  Current Word
                </p>
                <h2 className="text-6xl font-black italic transition-all duration-300">
                  {currentResult?.word || "..."}
                </h2>
                {currentResult && currentResult.confidence > 0 && (
                  <span className="inline-block mt-4 px-4 py-1 bg-white border border-pink-100 rounded-full text-pink-500 font-bold text-sm shadow-sm">
                    {(currentResult.confidence * 100).toFixed(1)}% Match
                  </span>
                )}
              </div>

              {/* AI 해석 결과 카드 (단어 나열 + Gemini 문장) */}
              <div className="max-w-2xl mx-auto p-8 bg-white/80 backdrop-blur-sm rounded-[40px] shadow-xl border border-white">
                <div className="mb-6 text-left">
                  <p className="text-[#330019]/40 text-[10px] font-black uppercase tracking-widest mb-2">
                    Word Sequence
                  </p>
                  <p className="text-xl font-bold text-[#330019]/70">
                    {currentResult?.sentence || "수어를 시작해 주세요."}
                  </p>
                </div>

                <div className="pt-6 border-t border-pink-50 text-left">
                  <p className="text-[#4AD799] text-[10px] font-black uppercase tracking-widest mb-2">
                    AI Interpretation (Gemini 3)
                  </p>
                  <p className="text-4xl font-black text-[#330019] leading-tight italic">
                    {currentResult?.translated ? (
                      `"${currentResult.translated}"`
                    ) : (
                      <span className="text-gray-200 italic">문장을 만드는 중...</span>
                    )}
                  </p>
                </div>
              </div>

              {/* 단어 인식 기록 로그 */}
              <HistoryLog logs={logs} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;