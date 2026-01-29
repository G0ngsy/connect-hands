interface Props {
  serverImage?: string; // 백엔드(Python)에서 웹소켓으로 보내줄 이미지 데이터
}

const CameraView = ({ serverImage }: Props) => {
  return (
    <div className="flex flex-col items-center w-full max-w-2xl mx-auto px-6 py-2">
      {/* 1. 메인 화면 컨테이너 */}
      <div className="relative w-full aspect-video bg-[#330019] rounded-[32px] overflow-hidden shadow-[0_20px_50px_rgba(51,0,25,0.2)] border-4 border-white">
        
        {/* 2. 화면 렌더링 (백엔드 영상 vs 대기 중) */}
        {serverImage ? (
          // 백엔드에서 이미지가 올 때 (Base64 이미지 출력)
          <img 
            src={`data:image/jpeg;base64,${serverImage}`} 
            className="w-full h-full object-cover" 
            alt="AI 피드"
          />
        ) : (
          // 데이터가 오기 전 대기 화면
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#330019]">
            <div className="w-10 h-10 border-4 border-pink-200 border-t-pink-500 rounded-full animate-spin mb-4"></div>
            <p className="text-white/40 text-sm font-medium">백엔드 카메라 연결 대기 중...</p>
          </div>
        )}

        {/* 3. 장식 오버레이 (디자인 요소) */}
        <div className="absolute top-6 left-6 flex items-center gap-2 bg-[#00A653]/80 backdrop-blur-md px-3 py-1 rounded-full border border-[#D6F5E6]/30">
          <span className="w-2 h-2 bg-[#D6F5E6] rounded-full animate-pulse"></span>
          <span className="text-white text-[10px] font-bold tracking-widest uppercase">LIVE AI TRACKING</span>
        </div>

        <div className="absolute bottom-6 right-6">
           <span className="text-[#FF66B2]/40 font-black italic text-sm uppercase">connect-hands</span>
        </div>
      </div>

      {/* 4. 하단 안내 캡션 */}
      <p className="mt-6 text-[#660033] text-sm font-medium opacity-60 flex items-center gap-2">
        <span className="w-1.5 h-1.5 bg-[#FF66B2] rounded-full"></span>
        수어 인식을 위해 손이 화면 안에 잘 들어오게 해주세요
      </p>
    </div>
  );
};

export default CameraView;