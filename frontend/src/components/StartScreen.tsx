
import { Video } from 'lucide-react'; // 아이콘 라이브러리 사용 권장

const StartScreen = ({ onStart }: { onStart: () => void }) => {
  return (
    <div className="flex flex-col items-center justify-center p-6">
      {/* 메인 카드: 부드러운 곡률과 깊이감 있는 그림자 */}
        <div className="w-full max-w-2xl bg-white rounded-[40px] p-12 
        shadow-[0_30px_100px_-20px_rgba(255,179,209,0.3),0_20px_50px_-10px_rgba(255,102,178,0.1)] 
        flex flex-col items-center text-center border border-white/60">
        
        {/* 카메라 아이콘 서클: Level_1 핑크 배경 */}
        <div className="w-24 h-24 bg-[#FFD6E6]/30 rounded-full flex items-center justify-center mb-8 border border-white shadow-inner">
          <Video size={40} className="text-[#FF66B2]" />
        </div>

        <h2 className="text-2xl font-bold text-[#330019] mb-3">실시간 수어 인식</h2>
        <p className="text-gray-400 text-sm leading-relaxed mb-10">
          카메라를 통해 당신의 손짓을<br />
          따뜻한 언어로 연결합니다.
        </p>

        {/* 시작 버튼: Level_4 ~ Level_5 그라데이션 및 인터랙션 */}
        <button 
          onClick={onStart}
          className="group flex items-center gap-3 bg-gradient-to-r from-[#FF66B2] to-[#FF3399] text-white px-8 py-4 rounded-2xl font-bold shadow-lg shadow-pink-200 hover:scale-105 transition-all duration-300 active:scale-95"
        >
          카메라 켜기
          <span className="group-hover:translate-x-1 transition-transform">→</span>
        </button>
      </div>
    </div>
  );
};

export default StartScreen;