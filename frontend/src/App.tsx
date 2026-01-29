
import Header from './components/Header';
import StartScreen from './components/StartScreen';
import HistoryLog from './components/HistoryLog';

function App() {
  return (
    // 전체 배경: 아주 연한 핑크톤 그라데이션으로 포근한 느낌 부여
    <div className="min-h-screen bg-gradient-to-b from-[#FFF5F8] to-[#FFFFFF] font-sans text-[#330019]">
      <Header />
      
      <main className="max-w-4xl mx-auto pt-2 pb-12 space-y-8">
        {/* 메인 섹션 */}
        <StartScreen onStart={() => console.log('Camera Start')} />
        
        {/* 로그 섹션 */}
        <HistoryLog />
      </main>
    </div>
  );
}

export default App;