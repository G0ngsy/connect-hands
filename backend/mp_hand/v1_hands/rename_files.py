import os

# 1. 파일이 저장된 경로 설정
data_path = r"C:\Users\akfnx\Desktop\suhwa\results"

# 2. 어떻게 바꿀지 설정 (예: 'HELLO'를 '안녕하세요'로 변경)
# 💡 여러 개를 바꾸고 싶다면 이 부분을 수정해서 여러 번 실행하세요.
target_word = "GOOD"      # 현재 파일명에 포함된 단어
replace_word = "좋다"  # 바꾸고 싶은 새 단어

def rename_data_files(path, old, new):
    if not os.path.exists(path):
        print(f"❌ 경로를 찾을 수 없습니다: {path}")
        return

    file_list = os.listdir(path)
    count = 0

    print(f"🔄 작업을 시작합니다: [{old}] -> [{new}]")

    for filename in file_list:
        # 파일명에 기존 단어가 포함되어 있고 확장자가 .npy인 것만 골라냅니다.
        if old in filename and filename.endswith('.npy'):
            old_full_path = os.path.join(path, filename)
            
            # 파일명에서 단어 교체
            new_filename = filename.replace(old, new)
            new_full_path = os.path.join(path, new_filename)
            
            # 실제 이름 변경
            try:
                os.rename(old_full_path, new_full_path)
                print(f" 성공: {filename} -> {new_filename}")
                count += 1
            except Exception as e:
                print(f" 실패: {filename} ({e})")

    print("-" * 30)
    print(f"✅ 총 {count}개의 파일명이 변경되었습니다.")

if __name__ == "__main__":
    rename_data_files(data_path, target_word, replace_word)