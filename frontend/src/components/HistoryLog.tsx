

const HistoryLog = () => {
  return (
    <div className="w-full max-w-2xl mx-auto px-6 mt-12 pb-10">
      {/* 
        🌸 메인 카드: 
        전체적으로 일관된 핑크빛 그림자와 화이트 톤을 유지하여 
        깨끗하고 정돈된 AI 서비스 이미지를 전달합니다.
      */}
      <div className="w-full bg-white rounded-[40px] overflow-hidden
        shadow-[0_30px_100px_-20px_rgba(255,179,209,0.3),0_20px_50px_-10px_rgba(255,102,178,0.1)] 
        border border-white/60">
        
        {/* 
          💡 헤더 영역 수정: 
          그라데이션을 빼고 'bg-white'로 통일하여 시각적 소음을 줄였습니다. 
        */}
        <div className="flex items-center justify-between px-8 py-5 border-b border-[#FFD6E6]/30 bg-white">
          <div className="flex items-center gap-3">
            {/* 민트 컬러 포인트 바 (Level_4: #4AD799) */}
            <span className="w-1.5 h-5 bg-[#4AD799] rounded-full shadow-[0_0_8px_rgba(74,215,153,0.3)]"></span>
            <span className="text-[#330019]/60 text-[12px] font-black uppercase tracking-[0.2em]">
              Recognized Words
            </span>
          </div>
          
          {/* 우측 장식용 도트: 핑크 팔레트로 포인트 */}
          <div className="flex gap-1.5">
             <div className="w-1.5 h-1.5 bg-[#FFB3D1] rounded-full"></div>
             <div className="w-1.5 h-1.5 bg-[#FFD6E6] rounded-full"></div>
          </div>
        </div>

        {/* 메인 콘텐츠 영역 */}
        <div className="relative p-12 flex flex-col items-center justify-center min-h-[220px] text-center bg-white">
          
          {/* 배경 미세 점 패턴: 0.02의 아주 낮은 불투명도로 고급스러움 유지 */}
          <div className="absolute inset-0 opacity-[0.02] pointer-events-none" 
               style={{ backgroundImage: 'radial-gradient(#FF007F 1px, transparent 1px)', backgroundSize: '20px 20px' }}>
          </div>

          {/* 텍스트 가이드 */}
          <h3 className="text-[#75E1B2] text-xl font-bold italic tracking-tight mb-8 relative z-10">
            손짓을 연결해 주세요
          </h3>
          
          {/* AI 스피너 애니메이션 (Level_4 핑크 포인트) */}
          <div className="relative w-14 h-14">
            <div className="absolute inset-0 border-[3px] border-[#FFD6E6] rounded-full"></div>
            <div className="absolute inset-0 border-[3px] border-t-[#FF66B2] border-r-transparent border-b-transparent border-l-transparent rounded-full animate-spin"></div>
            <div className="absolute inset-[18px] bg-[#FFB3D1]/40 rounded-full animate-pulse"></div>
          </div>
          
          {/* 하단 캡션 */}
          <p className="mt-8 text-[#330019]/20 text-[10px] font-bold tracking-widest uppercase">
            Waiting for input...
          </p>
        </div>
      </div>
    </div>
  );
};

export default HistoryLog;