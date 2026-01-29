// 1. 데이터 타입 정의
interface SignResult {
  word: string;
  confidence: number;
  is_detected: boolean;
  timestamp?: string;
}

// 2. Props 타입 정의
interface HistoryLogProps {
  logs: SignResult[];
}

const HistoryLog = ({ logs }: HistoryLogProps) => {
  return (
    <div className="w-full max-w-2xl mx-auto px-6 mt-12 pb-10">
      <div className="w-full bg-white rounded-[40px] overflow-hidden shadow-[0_30px_100px_-20px_rgba(255,179,209,0.3)] border border-white/60">
        
        {/* 헤더 영역 (고정) */}
        <div className="flex items-center justify-between px-8 py-5 border-b border-[#FFD6E6]/30 bg-white">
          <div className="flex items-center gap-3">
            <span className="w-1.5 h-5 bg-[#4AD799] rounded-full"></span>
            <span className="text-[#330019]/60 text-[12px] font-black uppercase tracking-[0.2em]">
              Recognized Words
            </span>
          </div>
        </div>

        {/* 메인 콘텐츠 영역 (조건부 렌더링) */}
        <div className="relative min-h-[220px] bg-white">
          
          {logs.length === 0 ? (
            /* ────────────────────────────────────────────────────────────
               A. 데이터가 없을 때: 대기 화면 (스피너)
            ───────────────────────────────────────────────────────────── */
            <div className="p-12 flex flex-col items-center justify-center text-center">
              <h3 className="text-[#75E1B2] text-xl font-bold italic mb-8">손짓을 연결해 주세요</h3>
              <div className="relative w-14 h-14">
                <div className="absolute inset-0 border-[3px] border-[#FFD6E6] rounded-full"></div>
                <div className="absolute inset-0 border-[3px] border-t-[#FF66B2] border-transparent rounded-full animate-spin"></div>
              </div>
              <p className="mt-8 text-[#330019]/20 text-[10px] font-bold tracking-widest uppercase">Waiting for input...</p>
            </div>
          ) : (
            /* ────────────────────────────────────────────────────────────
               B. 데이터가 있을 때: 실시간 로그 리스트
            ───────────────────────────────────────────────────────────── */
            <div className="p-6 space-y-3 max-h-[400px] overflow-y-auto custom-scrollbar">
              {logs.map((log, index) => (
                <div 
                  key={index} 
                  className="flex justify-between items-center p-5 bg-pink-50/30 rounded-3xl border border-pink-100/50 animate-in fade-in slide-in-from-top-2 duration-500"
                >
                  <div>
                    <p className="text-xl font-black text-[#330019] tracking-tighter">{log.word}</p>
                    <p className="text-[10px] text-gray-400 font-mono mt-1">{log.timestamp}</p>
                  </div>
                  <div className="text-right">
                    <span className="text-xs font-bold text-[#FF66B2] bg-white px-3 py-1 rounded-full border border-pink-100 shadow-sm">
                      {(log.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default HistoryLog;