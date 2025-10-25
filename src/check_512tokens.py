# check_token_length_nllb.py
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

# ===========================
# 설정
# ===========================
MODEL_NAME = "facebook/nllb-200-distilled-600M"
CSV_PATH = "src/Fake_News_Detection_Data_512.csv"  # CSV 경로
TEXT_COL = "text"                              # 텍스트 컬럼명
MAX_LEN = 512

# ===========================
# 1) 데이터 불러오기
# ===========================
df = pd.read_csv(CSV_PATH)
if TEXT_COL not in df.columns:
    TEXT_COL = df.columns[0]
texts = df[TEXT_COL].astype(str).tolist()

print(f"[INFO] Loaded {len(texts):,} samples")

# ===========================
# 2) 토크나이저 로드
# ===========================
tok = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="eng_Latn")

# ===========================
# 3) 토큰 길이 측정 (샘플링 가능)
# ===========================
SAMPLE_SIZE = min(44898, len(texts))  # 전체가 너무 크면 5000개만 확인
sample_texts = texts[:SAMPLE_SIZE]

enc = tok(sample_texts, truncation=True, padding=False, max_length=MAX_LEN)
lens = list(map(len, enc["input_ids"]))

# ===========================
# 4) 통계 출력
# ===========================
print("\n=== Token Length Stats ===")
print(f"Samples checked: {SAMPLE_SIZE}")
print(f"Min length : {np.min(lens)}")
print(f"Max length : {np.max(lens)}")
print(f"Mean length: {np.mean(lens):.2f}")
print(f"Median     : {np.median(lens)}")
print(f"90th pct   : {np.percentile(lens, 90)}")
print(f"99th pct   : {np.percentile(lens, 99)}")
print(f"100th pct  : {np.percentile(lens, 100)}")

# ===========================
# 5) 히스토그램 시각화
# ===========================
plt.figure(figsize=(8, 4))
plt.hist(lens, bins=50, color='skyblue', edgecolor='black')
plt.axvline(MAX_LEN, color='red', linestyle='--', label=f'MAX_LEN={MAX_LEN}')
plt.title("Token Length Distribution (NLLB tokenizer)")
plt.xlabel("Token length")
plt.ylabel("Number of samples")
plt.legend()
plt.tight_layout()
plt.show()
