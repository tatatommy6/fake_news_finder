import os
import pandas as pd
import torch
from transformers import MarianTokenizer, MarianMTModel
from tqdm import tqdm

MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-en-ko"

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    else: return torch.device("cpu")

def load_model(model_name=MODEL_NAME, device=None):
    tokenizer = MarianTokenizer.from_pretrained(model_name) #mariantokenizer: huggingface 번역 모델 전용 토크나이저
    model = MarianMTModel.from_pretrained(model_name)
    if device is None:
        device = get_device()
    model.to(device)
    model.eval() #평가모드로 전환
    return tokenizer, model, device

@torch.inference_mode()
def translate_batch(texts, tokenizer, model, device, max_src_len=512, max_tgt_len=512):
    # 배치 토크나이징
    enc = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_src_len,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    outs = model.generate(
        **enc,
        max_length=max_tgt_len,   # 번역 출력 최대 길이
        # num_beams=1,            # 기본값(빔서치 비활성). 품질 높이고 싶으면 4~6로
        # no_repeat_ngram_size=3, # 반복 억제 (optional)
    )
    return tokenizer.batch_decode(outs, skip_special_tokens=True) #토큰을 문장으로 디코딩하는 함수

def run_translation_pipeline(
    csv_path="/src/translated_subset.csv",
    output_path="/src/translated_korean.csv",
    text_col_candidates=("text", "truncated_text"),
    batch_size=16,
    n_limit=None 
):
    df = pd.read_csv(csv_path).head(10000)  # 상위 10,000줄만 사용
    print(f"Loaded {len(df)} rows for translation")

    # 0) 입력 확인
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    # 1) 텍스트 컬럼 자동 선택
    for c in text_col_candidates:
        if c in df.columns:
            text_col = c
            break
    else:
        raise KeyError(f"No text column among: {text_col_candidates}")

    # 2) 강제 상한(파일이 4만여 행이면 여기서 1만으로 제한)
    if n_limit is not None and len(df) > n_limit:
        df = df.sample(n=n_limit, random_state=42).reset_index(drop=True)
        print(f"[INFO] sampled to {len(df)} rows")

    texts = df[text_col].astype(str).tolist() # 10000

    # 3) 모델 로드
    model_name = MODEL_NAME
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device).eval()
    print(f"Device: {device}; Model: {model_name}")
    print(f"Translating {len(texts)} rows with batch_size={batch_size} ...")

    @torch.inference_mode()
    def translate_batch(batch):
        enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        outs = model.generate(**enc, max_length=512)
        return tokenizer.batch_decode(outs, skip_special_tokens=True)

    total = (len(texts)+batch_size-1)//batch_size
    translated = []
    for i in tqdm(range(0, len(texts), batch_size), total = total):
        translated.extend(translate_batch(texts[i:i+batch_size]))

    df["translated_text"] = translated
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Done. Saved to {output_path}")

if __name__ == "__main__":
    # 샘플링 단계 (문자자르기 제거 권장)
    # 원본: Fake_News_Detection_Data.csv 4만행 중 1만행 샘플
    src_path = "/src/Fake_News_Detection_Data_512.csv"
    subset_path = "/src/translated_korean.csv"

    if os.path.exists(src_path) and not os.path.exists(subset_path):
        sdf = pd.read_csv(src_path)
        # 문자 512 자르기 제거: 번역 품질 위해 원문 그대로
        sampled_df = sdf.sample(n=10000, random_state=42).copy()
        # 번역 후 한국어 모델에서 512 토큰 재분할 예정이므로 원문 유지
        # 필요한 컬럼만 저장
        keep_cols = []
        if "text" in sampled_df.columns:
            keep_cols.append("text")
        if "label" in sampled_df.columns:
            keep_cols.append("label")
        if not keep_cols:
            # 최소한 text만이라도 보장
            sampled_df["text"] = sampled_df.iloc[:, 0].astype(str) #
            keep_cols = ["text"]
        sampled_df[keep_cols].to_csv(subset_path, index=False)
        print(f"Sampled 10k to {subset_path}")

    # 번역 실행
    run_translation_pipeline(csv_path="src/Fake_News_Detection_Data_512.csv",
    output_path="src/translated_korean.csv",
    text_col_candidates=("text", "truncated_text"),
    batch_size=16,
    )