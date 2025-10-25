# 영어 -> 한국어 번역 스크립트
import os
import pandas as pd
import torch
import random
import pandas as pd, re, random
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# 기존 helsinki-nlp/opus-mt-tc-big-en-ko 의 성능이 많이 떨어지는 관계로 아래 모델로 교체
# 이 모델은 다국어 모델로 소스 언어와 타깃 언어를 명시해줘야함
MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"   # 입력 언어(영어)
TGT_LANG = "kor_Hang"   # 출력 언어(한국어, 한글 스크립트)
# generate 기본값 (품질 안정)
NUM_BEAMS = 1
MAX_SRC_LEN = 512
MAX_NEW_TOKENS = 512
NO_REPEAT_NGRAM_SIZE = 2
DO_SAMPLE = False       # 샘플링 금지 (중요)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_name=MODEL_NAME, src_lang=SRC_LANG, device=None):
    token = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = device or get_device()
    model.to(device).eval()
    print(f"Using device: {device}")
    return token, model, device

# --- 추가: 언어 ID 안전하게 얻는 도우미 형님 ---
def get_lang_id(token, lang_code: str) -> int:
    # 1. 최신 토크나이저에 존재
    if hasattr(token, "lang_code_to_id"):
        return token.lang_code_to_id[lang_code]
    # 1. Fast 토크나이저 일부 버전에 존재
    if hasattr(token, "get_lang_id"):
        return token.get_lang_id(lang_code)
    # 3. 토큰 문자열로 직접 변환 시도
    for cand in (lang_code, f"{lang_code}", f"__{lang_code}__"):
        tid = token.convert_tokens_to_ids(cand)
        if tid is not None and tid != token.unk_token_id:
            return tid
    raise ValueError(f"Cannot resolve language id for '{lang_code}'. Update transformers or tokenizer.")

@torch.inference_mode() # 이게뭐지
def translate_batch(
    texts,
    token,
    model,
    device,
    tgt_lang=TGT_LANG,
    max_src_len=MAX_SRC_LEN,
    max_new_tokens=MAX_NEW_TOKENS,
    # num_beams=NUM_BEAMS,
    # no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
    # do_sample=DO_SAMPLE,
):
    # 입력 정리(불필요 공백/개행 줄이기)
    texts = [str(t).strip() for t in texts]

    enc = token(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_src_len,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    
    forced_bos_token_id = get_lang_id(token, tgt_lang)
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
    return token.batch_decode(gen_ids, skip_special_tokens=True)

def run_translation_pipeline(csv_path, output_path, text_col_candidates=("text", "truncated_text"), batch_size=16,n_limit=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # 텍스트 컬럼 자동 선택
    for c in text_col_candidates:
        if c in df.columns:
            text_col = c
            break
    else:
        # 첫 컬럼을 텍스트로 강제 사용
        text_col = df.columns[0]
        print(f"No text column found in {text_col_candidates}. Using first column: {text_col}")

    texts = df[text_col].astype(str).tolist()

    token, model, device = load_model_and_tokenizer(MODEL_NAME, SRC_LANG)

    # === 길이 버킷 + 셔플 (패딩 낭비 down, 속도 출렁임↓) ===
    idxs = list(range(len(texts)))
    lens = [len(token.encode(str(texts[i]), add_special_tokens=True, truncation=True, max_length=MAX_SRC_LEN)) for i in idxs]
    pairs = sorted(zip(lens, idxs), key=lambda x: x[0])
    
    B = 16  # 버킷 수 (데이터 크기에 따라 8~32 권장)
    bucket_size = (len(pairs) + B - 1) // B
    buckets = [pairs[i:i+bucket_size] for i in range(0, len(pairs), bucket_size)]
    
    for b in buckets:
        random.shuffle(b)
    
    # 번역 실행 순서(원본 인덱스) 확정
    order = [idx for bucket in buckets for (_, idx) in bucket]

    enc_all = token(texts, return_tensors=None, truncation=True, padding=False, max_length=MAX_SRC_LEN)
    ids_all = enc_all["input_ids"]
    attn_all = enc_all["attention_mask"]

    # 아래는 번역 메인 루프와 그에 대한 설명
    # [성능 최적화 설명 — 사전 토큰화 + 패딩 분리]
    #
    # 기존 방식
    #   - 아래와 같은 for 루프 구조에서, 매 배치마다 token(batch, ...)을 호출함.
    #       for i in tqdm(...):
    #           batch = texts[i:i+batch_size]
    #           enc = token(batch, return_tensors="pt", truncation=True, padding=True, max_length=MAX_SRC_LEN)
    #
    #   - 이 구조는 매번 CPU가 토큰화를 새로 수행하기 때문에,
    #     GPU가 디코딩을 기다리는 동안 “놀게” 됨 → GPU 사용률이 5~60%대에서 들쭉날쭉함.
    #
    # 수정된 방식
    #   - 루프 진입 전에 **전체 텍스트를 한 번만 토큰화(token([...]))** 함.
    #   - 이렇게 얻은 input_ids / attention_mask 리스트를 enc_all에 캐시로 저장.
    #   - 이후 루프에서는 각 배치에 필요한 인덱스만 골라 pad(token.pad)하여 GPU로 전달.
    #   - 즉, 토큰화 과정(가장 무거운 CPU 작업)을 1회로 줄여 CPU 병목을 제거함.
    #
    # 효과
    #   - NllbTokenizerFast(빠른 토크나이저)는 내부적으로 C++ 병렬 토큰화를 사용하므로,
    #     encode()를 문장별로 부르는 것보다 token([...]) 한 번에 호출하는 게 훨씬 빠름.
    #   - GPU가 꾸준히 바쁘게 일하게 되어, GPU 사용률이 85~95% 근처로 안정됨.
    #   - s/it(배치당 처리 시간)의 편차가 줄어들고 ETA(예상 시간)가 안정화됨.
    #
    # 추가 설명
    #   - 루프에서는 pad만 수행하므로 enc = token.pad(...) 로 패딩만 맞춤.
    #   - 이렇게 하면 패딩 길이는 배치 내에서만 계산되므로 VRAM 효율도 좋아짐.
    #   - (NllbTokenizerFast 경고 “encode()+pad 대신 token([...]) 쓰라”도 자연스럽게 해결됨.)
    #
    # 요약
    #   “루프 안에서 매번 토큰화 → GPU가 논다” X
    #   “루프 전에 전체 토큰화 → 루프에서는 pad만 수행” OK

    total = (len(order) + batch_size - 1) // batch_size
    translated_slots = [""] * len(texts)  # 원래 자리수 만큼 확보
    
    translated_slots = [""] * len(texts)
    total = (len(order) + batch_size - 1) // batch_size

    # 메인 번역 루프
    for i in tqdm(range(0, len(order), batch_size), total = total, desc = "Translating"):
        batch_ids = order[i : i + batch_size]
        # 2) 배치마다 pad만 수행
        enc = token.pad( #.pad(): 여러 문장의 토큰 길이를 맞춰주는(=패딩하는) tokenizer가 제공하는 내장 함수
            {"input_ids": [ids_all[j] for j in batch_ids],
            "attention_mask": [attn_all[j] for j in batch_ids]},
            return_tensors="pt"
        )
        enc = {k : v.to(device, non_blocking = True) for k, v in enc.items()}
    
        with torch.autocast("cuda", dtype=torch.float16):
            out_ids = model.generate(
                **enc,
                forced_bos_token_id = get_lang_id(token, TGT_LANG),
                do_sample = False, num_beams = 1,
                max_new_tokens = MAX_NEW_TOKENS, use_cache=True
            )
        outs = token.batch_decode(out_ids, skip_special_tokens=True)
    
        for j, txt in zip(batch_ids, outs):
            translated_slots[j] = txt

    # === 저장 ===
    df["translated_text"] = translated_slots
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Done. Saved to {output_path}")


if __name__ == "__main__":
    # 사용 예시
    run_translation_pipeline(
        csv_path = "retranslate_candidates.csv",
        output_path = "retranslated_fixed.csv",
        text_col_candidates = ("text", "truncated_text"),
        batch_size = 80, #16 -> 32 -> 64 -> 80(최종. 128은 안됨)
        n_limit = None,
    )


    # --- 번역 결과 검수 ---
    df = pd.read_csv("retranslated_fixed.csv")
    # 1. 언어감지 (샘플만)
    sample_idx = random.sample(range(len(df)), k=min(500, len(df)))
    ko_ok = sum(1 for i in sample_idx if (df.loc[i,"translated_text"] and detect(df.loc[i,"translated_text"])=="ko"))
    print("KO rate (sample):", ko_ok/len(sample_idx))
    
    # 2. 길이비율
    def ratio_ok(en, ko):
        return 0.6 <= (len(ko)+1)/(len(en)+1) <= 1.6
    df["len_ratio_ok"] = [ratio_ok(str(e), str(k)) for e,k in zip(df["text"], df["translated_text"])]
    
    # 3. 고정문구/쿠키배너 (솔직히 필요 없을거 같은데 일단 넣고)
    bad_patterns = ["이 웹 사이트는 쿠키", "개인 정보 보호 정책", "동의하지 않으면 웹 사이트를 떠나십시오"]
    patterns = re.compile("|".join(bad_patterns))
    df["has_bad"] = df["translated_text"].fillna("").str.contains(patterns)
    
    flagged = df[ (~df["len_ratio_ok"]) | (df["has_bad"]) ]
    print("Flagged rows:", len(flagged))
    flagged.to_csv("translated_flagged.csv", index=False)


    # --- 잘리거나 미완성된 번역 기사를 재번역할 후보 추출 ---
    print("Checking for truncated or incomplete translations...")
    
    token = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)
    
    def gen_len(s):
        # 번역된 문장의 토큰 길이 측정
        return len(token(str(s), truncation=True, padding=False, max_length=MAX_SRC_LEN)["input_ids"])
    
    # 1. 토큰 길이가 MAX_NEW_TOKENS 근처인 문장 (잘렸을 가능성 높은 관계로)
    df["gen_len"] = df["translated_text"].fillna("").map(gen_len)
    flag_len = df["gen_len"] >= (MAX_NEW_TOKENS - 4)
    
    # 2. 문장 끝이 마무리되지 않은 경우 (문장 중간에서 끊기거나 이상한곳에 종결점이 찍힌 경우)
    is_end_ok = df["translated_text"].fillna("").str.strip().str.endswith(tuple([".", "!", "?", "요.", "니다.", "다."]))
    flag_end = ~is_end_ok
    
    # 3. 둘 중 하나라도 해당하면 재번역 후보
    flag = flag_len | flag_end
    candidates = df[flag]
    
    print(f"Number of articles to be retranslated: {len(candidates)} / {len(df)}")
    candidates.to_csv("retranslate_candidates.csv", index = False, encoding = "utf-8")
    print("Saved to retranslate_candidates.csv")