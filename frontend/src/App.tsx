import { useState, useEffect } from 'react';
import Header from './components/Header';
import StartScreen from './components/StartScreen';
import HistoryLog from './components/HistoryLog';
import CameraView from './components/CameraView';

interface SignResult {
  word: string;
  confidence: number;
  is_detected: boolean;
  image?: string; 
  timestamp?: string;
}

function App() {
  const [isStarted, setIsStarted] = useState(false);
  const [currentResult, setCurrentResult] = useState<SignResult | null>(null);
  const [logs, setLogs] = useState<SignResult[]>([]);

  useEffect(() => {
    
    let socket: WebSocket | null = null;

    if (isStarted) {
      // [중요] 백엔드 서버 주소와 포트 확인 (ws:// 로 시작)
      socket = new WebSocket('ws://127.0.0.1:8080/ws');

      socket.onopen = () => console.log("✅ 서버 연결 성공!");

      socket.onmessage = (event) => {
        const data: SignResult = JSON.parse(event.data);
        setCurrentResult(data);

        // 90% 이상 일치할 때만 기록에 저장
        if (data.confidence > 0.90) {
          setLogs((prev) => {
            if (prev.length > 0 && prev[0].word === data.word) return prev;
            const newEntry = {
              ...data,
              timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
            };
            return [newEntry, ...prev].slice(0, 10);
          });
        }
      };

      socket.onerror = (err) => {
        console.error("❌ 연결 에러 발생:", err);
      };

      socket.onclose = () => console.log("ℹ️ 서버와 연결이 종료되었습니다.");
    }

    return () => socket?.close();
  }, [isStarted]);

  return (
    <div className="min-h-screen bg-[#FFF5F8] font-sans text-[#330019]">
      <Header />
      <main className="max-w-4xl mx-auto pt-2 pb-12 space-y-8">
        {!isStarted ? (
          <StartScreen onStart={() => setIsStarted(true)} />
        ) : (
          <div className="space-y-8">
            {/* 백엔드에서 받은 Base64 이미지를 컴포넌트에 전달 */}
            <CameraView serverImage={currentResult?.image} />
            
            <div className="text-center">
              <p className="text-[#FF66B2] font-bold text-xs tracking-widest uppercase mb-2">AI Recognition</p>
              <h2 className="text-6xl font-black">{currentResult?.word || "..."}</h2>
              {currentResult && currentResult.confidence > 0 && (
                <span className="inline-block mt-4 px-4 py-1 bg-white border border-pink-100 rounded-full text-pink-500 font-bold">
                  {(currentResult.confidence * 100).toFixed(1)}% Match
                </span>
              )}
            </div>
          </div>
        )}
        <HistoryLog logs={logs} />
      </main>
    </div>
  );
}

export default App;