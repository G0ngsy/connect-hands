

const Header = () => {
  return (
    <header className="flex justify-between items-center px-8 py-4 bg-white/80 backdrop-blur-md sticky top-0 z-50 border-b border-pink-50/50">
      
      {/* 
        로고 영역 전체 컨테이너: 
        아이콘과 텍스트가 서로 간섭하지 않도록 gap을 충분히 주고 items-center로 정렬 
      */}
      <div className="flex items-center gap-5 group cursor-pointer h-12">
        
        {/* 
          1. 아이콘 박스 (고정 영역):
          - w-20을 주어 👉👈가 움직이거나 [CH]로 변해도 텍스트 위치가 고정되게 함.
          - overflow-visible을 설정해 애니메이션 효과가 박스 밖으로 살짝 나가도 잘리지 않게 함.
        */}
        <div className="relative w-20 h-12 flex items-center justify-center overflow-visible flex-shrink-0">
          
          {/* 왼쪽 손가락 (👉) */}
          <div className="absolute left-0 text-3xl transition-all duration-500 ease-in-out 
            group-hover:translate-x-6 group-hover:opacity-0 group-hover:scale-50">
            <span style={{ color: '#00CC66' }}>👉</span>
          </div>

          {/* 오른쪽 손가락 (👈) */}
          <div className="absolute right-0 text-3xl transition-all duration-500 ease-in-out 
            group-hover:-translate-x-6 group-hover:opacity-0 group-hover:scale-50">
            <span style={{ color: '#FF3399' }}>👈</span>
          </div>

          {/* 변신하는 [CH] 사각형 */}
          <div className="absolute inset-0 flex items-center justify-center opacity-0 scale-0 
            group-hover:opacity-100 group-hover:scale-100 transition-all duration-500 delay-100">
            
            <div className="w-11 h-11 bg-gradient-to-br from-[#FF66B2] to-[#4AD799] 
              rounded-xl flex items-center justify-center shadow-lg shadow-pink-200/50 
              transform group-hover:rotate-[360deg] transition-transform duration-700">
              <span className="text-white font-black text-sm tracking-tighter">CH</span>
            </div>

            {/* 임팩트 효과 (라임 컬러) */}
            <div className="absolute inset-0 bg-[#A3D14A] rounded-full blur-xl opacity-0 
              group-hover:animate-[ping_1.2s_ease-out_infinite] pointer-events-none"></div>
          </div>
        </div>

        {/* 
          2. 텍스트 로고 영역 (고정 위치):
          - whitespace-nowrap: 글자가 아래로 떨어지거나 잘리는 것을 절대 방어
          - flex-shrink-0: 아이콘이 커져도 텍스트가 압축되지 않음
        */}
        <div className="flex flex-col whitespace-nowrap flex-shrink-0">
          <h1 className="text-[#330019] font-black leading-none text-xl tracking-tighter">
            connect<span className="text-[#FF66B2]">·</span>hands
          </h1>
          <div className="flex items-center gap-1.5 mt-1.5">
            {/* 세련된 민트 라인 장식 */}
            <span className="w-4 h-[2px] bg-[#00CC66] rounded-full"></span>
            <p className="text-[#00CC66] text-[10px] font-black tracking-[0.25em] uppercase leading-none">
              Sign Language AI
            </p>
          </div>
        </div>
      </div>

      {/* 3. 우측 상태 배지 (READY) */}
      <div className="flex-shrink-0 ml-4 flex items-center gap-2 bg-[#D6F5E6] px-4 py-1.5 rounded-xl border border-[#A3EBCC] shadow-sm">
        <span className="w-2 h-2 bg-[#00CC66] rounded-full animate-pulse shadow-[0_0_8px_#00CC66]"></span>
        <span className="text-[#006633] text-[11px] font-black tracking-widest uppercase">Ready</span>
      </div>
    </header>
  );
};

export default Header;