// [Props 정의] onGoHome 함수를 부모(App.tsx)로부터 받습니다.
interface HeaderProps {
  onGoHome: () => void;
}

const Header = ({ onGoHome }: HeaderProps) => {
  return (
    <header className="flex justify-between items-center px-8 py-4 bg-white/80 backdrop-blur-md sticky top-0 z-50 border-b border-pink-50/50">
      
      {/* 
        [핵심] 클릭 시 onGoHome(setIsStarted(false))을 실행합니다. 
      */}
      <div 
        onClick={onGoHome} 
        className="flex items-center gap-5 group cursor-pointer h-12"
      >
        <div className="relative w-20 h-12 flex items-center justify-center overflow-visible flex-shrink-0">
          
          {/* 애니메이션 손가락 (👉 👈) */}
          <div className="absolute left-0 text-3xl transition-all duration-500 ease-in-out group-hover:translate-x-6 group-hover:opacity-0 group-hover:scale-50">
            <span style={{ color: '#00CC66' }}>👉</span>
          </div>
          <div className="absolute right-0 text-3xl transition-all duration-500 ease-in-out group-hover:-translate-x-6 group-hover:opacity-0 group-hover:scale-50">
            <span style={{ color: '#FF3399' }}>👈</span>
          </div>

          {/* 클릭 가능한 [CH] 로고 박스 */}
          <div className="absolute inset-0 flex items-center justify-center opacity-0 scale-0 group-hover:opacity-100 group-hover:scale-100 transition-all duration-500 delay-100">
            <div className="w-11 h-11 bg-gradient-to-br from-[#FF66B2] to-[#4AD799] rounded-xl flex items-center justify-center shadow-lg transform group-hover:rotate-[360deg] transition-transform duration-700">
              <span className="text-white font-black text-sm">CH</span>
            </div>
            {/* 파동 효과 */}
            <div className="absolute inset-0 bg-[#A3D14A] rounded-full blur-xl opacity-0 group-hover:animate-ping pointer-events-none"></div>
          </div>
        </div>

        {/* 텍스트 로고 영역 */}
        <div className="flex flex-col whitespace-nowrap">
          <h1 className="text-[#330019] font-black leading-none text-xl tracking-tighter">
            connect<span className="text-[#FF66B2]">·</span>hands
          </h1>
          <p className="text-[#00CC66] text-[10px] font-black tracking-[0.25em] mt-1.5 uppercase">Sign Language AI</p>
        </div>
      </div>

   
    </header>
  );
};

export default Header;