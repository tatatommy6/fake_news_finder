#영어 데이터셋은 구했지만 한글 모델을 만들고 싶어 4만줄 중 1만줄만 한국어로 번역해서 사용하고 싶어 샘플링 하는 코드
#샘플링 한 데이터를 huggingface의 영어-> 한국어 번역 모델을 써서 mps+ 벙렬처리로 빠르게 번역할 예정

#모델 다운로드 하는 부분
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-ko")
# model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-tc-big-en-ko")

# 샘플링하는 부분
# import pandas as pd
# # 실제 파일 경로로 수정해야 함
# file_path = "Fake_News_Detection_Data.csv"

# # CSV 불러오기
# df = pd.read_csv(file_path)

# # 1만 줄 랜덤 샘플링
# sampled_df = df.sample(n=10000, random_state=42).copy()

# # 최대 512자까지 텍스트 자르기
# sampled_df['truncated_text'] = sampled_df['text'].apply(lambda x: str(x)[:512])

# # 결과 저장 (원한다면)
# sampled_df[['truncated_text', 'label']].to_csv("translated_subset.csv", index=False)



#모델을 사용해서 번역하는 부분
import pandas as pd
import torch
from transformers import MarianTokenizer, MarianMTModel
from multiprocessing import Pool, cpu_count
import os

# 모델 및 토크나이저 로드
model_name = "Helsinki-NLP/opus-mt-tc-big-en-ko"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# MPS 디바이스 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

# 번역 함수 (문장 하나 처리)
def translate_line(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="longest", max_length=512).to(device)
        with torch.no_grad():
            translated = model.generate(**inputs, max_length=512)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        return result
    except Exception as e:
        return f"[ERROR] {str(e)}"

# 병렬 처리용 래퍼 함수
def translate_lines(text_list, num_processes=4):
    with Pool(processes=num_processes) as pool:
        results = pool.map(translate_line, text_list)
    return results

# CSV 불러오기 및 번역 실행
def run_translation_pipeline(csv_path="translated_subset.csv", output_path="translated_korean.csv"):
    df = pd.read_csv(csv_path)
    texts = df['truncated_text'].astype(str).tolist()

    # 병렬 번역 실행 (CPU 코어 수 기준)
    num_proc = min(cpu_count(), 4)  # 4코어도 빡셈... 포기
    print(f"Using {num_proc} processes for translation...")
    translated = translate_lines(texts, num_processes=num_proc)

    # 결과 저장
    df['translated_text'] = translated
    df.to_csv(output_path, index=False)
    print(f"Translation complete! Saved to {output_path}")

# 실행
# mulitiprocessing을 사용할 때는 스크립트를 한번더 임포트 하는데 이때 __main__이 없으면
# 무한 재귀가 발행하므로 __main__을 사용해야함
# run_translation_pipeline("translated_subset.csv", "translated_korean.csv") # 단독 X

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    run_translation_pipeline("translated_subset.csv", "translated_korean.csv")
