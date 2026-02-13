import { useState, useEffect } from 'react';
import Header from './components/Header';
import StartScreen from './components/StartScreen';
import HistoryLog from './components/HistoryLog';
import CameraView from './components/CameraView';

// 공통 데이터 타입 정의
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
  const [isStarted, setIsStarted] = useState(false); // 카메라 켜짐 여부
  const [currentResult, setCurrentResult] = useState<SignResult | null>(null); // 실시간 결과
  const [logs, setLogs] = useState<SignResult[]>([]); // 단어 히스토리 로그

  useEffect(() => {
    let socket: WebSocket | null = null;
    if (isStarted) {
      // 카메라가 켜졌을 때만 실시간 웹소켓 연결 시작
      // socket = new WebSocket('ws://127.0.0.1:8080/ws');
      socket = new WebSocket('wss://connect-hands-api.onrender.com/ws');
      socket.onmessage = (event) => {
        const data: SignResult = JSON.parse(event.data);
        setCurrentResult(data);

        // 신뢰도가 높고 유효한 단어일 때만 로그에 저장
        if (data.confidence > 0.90 && data.word !== '' && data.word !== 'IDLE') {
          setLogs((prev) => {
            if (prev.length > 0 && prev[0].word === data.word) return prev;
            return [{
              ...data,
              timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            }, ...prev].slice(0, 10);
          });
        }
      };
    }
    return () => socket?.close();
  }, [isStarted]);

  return (
    <div className="min-h-screen bg-[#FFF5F8] font-sans text-[#330019]">
      {/* 로고 클릭 시 홈(시작 화면)으로 돌아가도록 onGoHome 함수 전달 */}
      <Header onGoHome={() => setIsStarted(false)} />

      <main className="max-w-4xl mx-auto pt-2 pb-12 space-y-8">
        {!isStarted ? (
          /* [1] 홈 화면: 로그 없이 깔끔하게 시작 버튼만 노출 */
          <StartScreen onStart={() => setIsStarted(true)} />
        ) : (
          /* [2] 실시간 인식 화면: 카메라가 켜지면 결과 카드와 로그가 함께 나타남 */
          <div className="space-y-8 animate-in fade-in duration-700">
            <CameraView serverImage={currentResult?.image} />
            
            <div className="text-center space-y-6">
              {/* 현재 단어 표시부 */}
              <div>
                <p className="text-[#FF66B2] font-bold text-xs uppercase mb-2">Current Word</p>
                <h2 className="text-6xl font-black italic">{currentResult?.word || "..."}</h2>
                {currentResult && currentResult.confidence > 0 && (
                  <span className="px-4 py-1 bg-white border border-pink-100 rounded-full text-pink-500 font-bold text-sm">
                    {(currentResult.confidence * 100).toFixed(1)}% Match
                  </span>
                )}
              </div>

              {/* AI 문장 해석 카드 */}
              <div className="max-w-2xl mx-auto p-8 bg-white/80 rounded-[40px] shadow-xl border border-white">
                <div className="mb-6 text-left">
                  <p className="text-[#330019]/40 text-[10px] font-black uppercase mb-2">Word Sequence</p>
                  <p className="text-xl font-bold text-[#330019]/70">{currentResult?.sentence || "수어를 시작하세요"}</p>
                </div>
                <div className="pt-6 border-t border-pink-50 text-left">
                  <p className="text-[#4AD799] text-[10px] font-black uppercase mb-2">AI Interpretation (Gemini)</p>
                  <p className="text-4xl font-black text-[#330019] leading-tight italic">
                    {currentResult?.translated ? `"${currentResult.translated}"` : "..."}
                  </p>
                </div>
              </div>

              {/* [3] 히스토리 로그: 인식 화면 내 하단에 위치하여 사용성 개선 */}
              <HistoryLog logs={logs} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;