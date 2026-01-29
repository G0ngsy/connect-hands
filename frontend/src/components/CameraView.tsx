import  { useEffect, useRef } from 'react';

const CameraView = () => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const setupCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720 },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch { 
        // 💡 해결: 'err'를 생략하여 ESLint 에러를 방지합니다.
        alert("카메라 권한을 허용해주세요!");
      }
    };

    setupCamera();
  }, []);

  return (
    <div className="flex flex-col items-center w-full max-w-2xl mx-auto px-6 py-2">
      {/* 카메라 컨테이너: 깊이감 있는 그림자와 부드러운 테두리 */}
      <div className="relative w-full aspect-video bg-[#330019] rounded-[32px] overflow-hidden shadow-[0_20px_50px_rgba(51,0,25,0.2)] border-4 border-white">
        
        {/* 실제 비디오 엘리먼트 */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          className="w-full h-full object-cover transform scale-x-[-1]" // 좌우 반전(미러 모드)
        />

        {/* 상단 오버레이: LIVE 표시 (Mint Level_5 & Level_6 활용) */}
        <div className="absolute top-6 left-6 flex items-center gap-2 bg-[#00A653]/80 backdrop-blur-md px-3 py-1 rounded-full border border-[#D6F5E6]/30">
          <span className="w-2 h-2 bg-[#D6F5E6] rounded-full animate-pulse"></span>
          <span className="text-white text-[10px] font-bold tracking-widest">LIVE AI TRACKING</span>
        </div>

        {/* 하단 오버레이: 가이드 라인 (디자인 포인트) */}
        <div className="absolute inset-0 border-[20px] border-white/5 pointer-events-none"></div>
        <div className="absolute bottom-6 right-6">
           {/* Level_4 핑크를 활용한 은은한 로고 워터마크 */}
           <span className="text-[#FF66B2]/40 font-black italic text-sm">connect-hands</span>
        </div>
      </div>

      {/* 카메라 하단 캡션 */}
      <p className="mt-6 text-[#660033] text-sm font-medium opacity-60 flex items-center gap-2">
        <span className="w-1.5 h-1.5 bg-[#FF66B2] rounded-full"></span>
        수어 인식을 위해 손이 화면 안에 잘 들어오게 해주세요
      </p>
    </div>
  );
};

export default CameraView;