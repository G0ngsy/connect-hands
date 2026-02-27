
// App.tsx와 타입을 일치시켜 에러 방지
interface SignResult {
  word: string;
  confidence: number;
  sentence: string;
  translated: string[];
  image: string;
  is_detected?: boolean;
  timestamp?: string;
}

interface HistoryLogProps {
  logs: SignResult[];
}

const HistoryLog = ({ logs }: HistoryLogProps) => {
  return (
    <div className="w-full max-w-2xl mx-auto px-6 mt-6">
      <div className="w-full bg-white rounded-[40px] overflow-hidden shadow-2xl border border-white/60">
        
        {/* 로그 헤더바 */}
        <div className="flex items-center px-8 py-5 border-b border-pink-50 bg-white">
          <div className="flex items-center gap-3">
            <span className="w-1.5 h-5 bg-[#4AD799] rounded-full"></span>
            <span className="text-[#330019]/60 text-[12px] font-black uppercase tracking-[0.2em]">Recognized Words Log</span>
          </div>
        </div>

        {/* 로그 데이터 리스트 */}
        <div className="min-h-[150px] bg-white">
          {logs.length === 0 ? (
            /* 데이터 없을 때의 스피너 애니메이션 */
            <div className="p-10 flex flex-col items-center justify-center">
              <div className="w-10 h-10 border-4 border-pink-100 border-t-pink-500 rounded-full animate-spin"></div>
              <p className="mt-4 text-[#330019]/20 text-[10px] font-bold uppercase tracking-widest">Waiting for input...</p>
            </div>
          ) : (
            /* 데이터가 있을 때의 스크롤 리스트 */
            <div className="p-6 space-y-3 max-h-[300px] overflow-y-auto">
              {logs.map((log, index) => (
                <div key={index} className="flex justify-between items-center p-4 bg-pink-50/30 rounded-3xl border border-pink-100/50">
                  <div className="text-left">
                    {/* 단어 대신 사용자가 선택한 문장을 보여줍니다. */}
                    <p className="text-lg font-black text-[#330019]">{log.translated[0]}</p>
                    {/* 보조 정보로 인식된 마지막 단어와 시간을 작게 표시합니다. */}
                    <p className="text-[10px] text-gray-400 mt-1"> {log.timestamp} • Last Word: {log.word}</p>
                  </div>
                  {/* <span className="text-xs font-bold text-pink-500 bg-white px-3 py-1 rounded-full shadow-sm">
                    {(log.confidence * 100).toFixed(0)}%
                  </span> */}
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