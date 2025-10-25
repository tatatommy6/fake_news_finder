# translate_nllb_en2ko.py
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# 기존 helsinki-nlp/opus-mt-tc-big-en-ko 의 성능이 많이 떨어지는 관계로 아래 모델로 교체
# 이 모델은 다국어 모델로 소스 언어와 타깃 언어를 명시해줘야함
MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"   # 입력 언어(영어)
TGT_LANG = "kor_Hang"   # 출력 언어(한국어, 한글 스크립트)

# generate 기본값 (품질 안정)
NUM_BEAMS = 5
MAX_SRC_LEN = 512
MAX_NEW_TOKENS = 256
NO_REPEAT_NGRAM_SIZE = 3
DO_SAMPLE = False       # 샘플링 금지 (중요)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_name=MODEL_NAME, src_lang=SRC_LANG, device=None):
    tok = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = device or get_device()
    model.to(device).eval()
    print(f"[INFO] Using device: {device}")
    return tok, model, device

# --- 추가: 언어 ID 안전하게 얻는 헬퍼 ---
def get_lang_id(tok, lang_code: str) -> int:
    # 1) 최신 토크나이저에 존재
    if hasattr(tok, "lang_code_to_id"):
        return tok.lang_code_to_id[lang_code]
    # 2) Fast 토크나이저 일부 버전에 존재
    if hasattr(tok, "get_lang_id"):
        return tok.get_lang_id(lang_code)
    # 3) 토큰 문자열로 직접 변환 시도 (버전에 따라 표기가 다를 수 있음)
    for cand in (lang_code, f"<<{lang_code}>>", f"__{lang_code}__"):
        tid = tok.convert_tokens_to_ids(cand)
        if tid is not None and tid != tok.unk_token_id:
            return tid
    raise ValueError(f"Cannot resolve language id for '{lang_code}'. Update transformers or tokenizer.")


@torch.inference_mode()
def translate_batch(
    texts,
    tok,
    model,
    device,
    tgt_lang=TGT_LANG,
    max_src_len=MAX_SRC_LEN,
    max_new_tokens=MAX_NEW_TOKENS,
    num_beams=NUM_BEAMS,
    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
    do_sample=DO_SAMPLE,
):
    # 입력 정리(불필요 공백/개행 줄이기)
    texts = [str(t).strip() for t in texts]

    enc = tok(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_src_len,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    
    forced_bos_token_id = get_lang_id(tok, tgt_lang)
    gen_ids = model.generate(
    **enc,
    forced_bos_token_id=forced_bos_token_id,
    do_sample=False,
    num_beams=5,
    no_repeat_ngram_size=3,
    max_new_tokens=256,
    early_stopping=True,
    use_cache=True,
    )
    return tok.batch_decode(gen_ids, skip_special_tokens=True)

def run_translation_pipeline(
    csv_path,
    output_path,
    text_col_candidates=("text", "truncated_text"),
    batch_size=16,
    n_limit=None,                 # 예: 10000 → 상위 1만 줄만 번역
    random_sample=False,          # True면 무작위 샘플링
    ):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded {len(df)} rows from {csv_path}")

    # 텍스트 컬럼 자동 선택
    for c in text_col_candidates:
        if c in df.columns:
            text_col = c
            break
    else:
        # 첫 컬럼을 텍스트로 강제 사용
        text_col = df.columns[0]
        print(f"[WARN] No text column found in {text_col_candidates}. Using first column: {text_col}")

    # 제한 적용
    if n_limit is not None and len(df) > n_limit:
        if random_sample:
            df = df.sample(n=n_limit, random_state=42).reset_index(drop=True)
            print(f"[INFO] Random-sampled to {len(df)} rows")
        else:
            df = df.head(n_limit).reset_index(drop=True)
            print(f"[INFO] Head-limited to {len(df)} rows")

    texts = df[text_col].astype(str).tolist()

    tok, model, device = load_model_and_tokenizer(MODEL_NAME, SRC_LANG)
    
    lens = [len(tok.encode(str(t), add_special_tokens=True, truncation=True, max_length=MAX_SRC_LEN))
        for t in texts]
    texts = [x for _, x in sorted(zip(lens, texts), key=lambda x: x[0])]

    total = (len(texts) + batch_size - 1) // batch_size
    translated = []
    for i in tqdm(range(0, len(texts), batch_size), total=total, desc="Translating"):
        batch = texts[i : i + batch_size]
        out = translate_batch(batch, tok, model, device)
        translated.extend(out)

    df["translated_text"] = translated
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[INFO] Done. Saved to {output_path}")

if __name__ == "__main__":
    # 사용 예시
    run_translation_pipeline(
        csv_path="Fake_News_Detection_Data_512.csv",
        output_path="translated_korean.csv",
        text_col_candidates=("text", "truncated_text"),
        batch_size=32,
        n_limit=None,        # ← 여기 숫자 바꿔서 1만 줄만 먼저 번역
        random_sample=False,  # True로 바꾸면 무작위 1만 줄
    )
    import pandas as pd, re, random
    from langdetect import detect
    
    df = pd.read_csv("translated_korean.csv")
    
    # 1) 언어감지 (샘플만)
    sample_idx = random.sample(range(len(df)), k=min(500, len(df)))
    ko_ok = sum(1 for i in sample_idx if (df.loc[i,"translated_text"] and detect(df.loc[i,"translated_text"])=="ko"))
    print("KO rate (sample):", ko_ok/len(sample_idx))
    
    # 2) 길이비율
    def ratio_ok(en, ko):
        return 0.6 <= (len(ko)+1)/(len(en)+1) <= 1.6
    df["len_ratio_ok"] = [ratio_ok(str(e), str(k)) for e,k in zip(df["text"], df["translated_text"])]
    
    # 3) 고정문구/쿠키배너
    bad_patterns = ["이 웹 사이트는 쿠키", "개인 정보 보호 정책", "동의하지 않으면 웹 사이트를 떠나십시오"]
    pat = re.compile("|".join(bad_patterns))
    df["has_bad"] = df["translated_text"].fillna("").str.contains(pat)
    
    flagged = df[ (~df["len_ratio_ok"]) | (df["has_bad"]) ]
    print("Flagged rows:", len(flagged))
    flagged.to_csv("translated_flagged.csv", index=False)