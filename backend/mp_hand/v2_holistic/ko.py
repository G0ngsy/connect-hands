import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# 1. 모델과 토크나이저 로드 (처음 실행 시 다운로드 시간이 몇 분 걸립니다)
print("🚀 모델 로딩 중... 잠시만 기다려주세요.")
tokenizer = PreTrainedTokenizerFast.from_pretrained('hyunwoongko/kobart')
model = BartForConditionalGeneration.from_pretrained('hyunwoongko/kobart')

# 2. 입력 단어 설정
input_text = "나 너 만나다 반갑다"
print(f"\n입력 단어: {input_text}")

# 3. 토큰화 및 문장 생성
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(
    inputs['input_ids'], 
    max_length=50, 
    num_beams=5, 
    epsilon_cutoff=0.01,
    early_stopping=True
)

# 4. 결과 출력
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"✨ AI 결과: {result}")