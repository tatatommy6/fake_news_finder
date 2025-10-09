"""
기사 전체를 요약한 뒤 최종 텍스트를 타깃 토크나이저 기준 512 토큰 이하로 강제 압축하는 코드
input: csv 최소['text'](option:['label','id','title'])
output: csv 최소['text','label'(있으면 유지), 'id'(있으면 유지)]

pipline:
1. 매우 긴 원문 -> 1024 토큰 단위로 청크 분할
2. 각 청크를 모델별로 요약
3. 요약문들을 이어붙임 -> 근데 여전히 길면 재요약 함
4. 재요약을 하여 압축된 버전 완성 -> 최종적으로 512토큰 기준으로 잘라냄

caution : only for ENG text

왜 처음부터 512로 요약하지 않았나
1. 요약 모델의 입출력 구조 차이 때문
    -사용할 facebook/bart-large-cnn 모델은 입력 문서 길이에 따라 요약 비율을 동적으로 결정함.

    -예를 들어 1024토큰 짜리 문서를 넣으면 그중 핵심 20~30%를 뽑은데 
    이미 출력 최대 길이를 너무 짧게 제한해버리면

    - 모델이 문서의 전반적인 구조를 충분히 보지 못하여 핵심을 빼먹은채 초반만 요약해버릴 수 있음.

2. 긴 문서일수록 요약의 맥락 유지가 중요함.
    - 뉴스 기사 1개가 2천 ~ 3천 토큰이면 처음부터 512토큰으로 자르면 모델은 전체 문맥을 못보고
    문단간 인과관계, 배경 설명등을 버림

    - 반면 한번은 넉넉히 보고(1차) -> 그걸 압축(2차)하는게 핵심 유지율이 높다고 판단함.
"""


import os
import math
import argparse
import re
import pandas as pd
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


#utils
def split_into_chunks_by_tokens(text:str , tokenizer, max_tokens: int = 1024, stride: int = 128)-> List[str]:
    #요약 모델 한계 내로 텍스트를 겹치게 잘라서 청크 리스트로 변환
    if not isinstance(text, str):
        return []
    ids = tokenizer.encode(text, add_special_tokens=False) # 문서를 토크나이저로 변환한 숫자 토큰들의 리스트
    if len(ids) <= max_tokens:
        return [text]
    
    chunks = []
    start = 0
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        sub_ids = ids[start:end]
        chunk = tokenizer.decode(sub_ids, skip_special_tokens=True)
        chunks.append(chunk)
        if end == len(ids):
            break
        start = end - stride
        if start < 0:
            start = 0
        if start >= len(ids):
            break
    return chunks


def sentence(text:str )-> List[str]:
    # 단순 영문 문장 불리기 (뉴스에 적합한 규칙 기반)
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text) #안에는 re.split()으로 분리된 문장 후보들이 있음
    return [s.strip() for s in sentences if s.strip()] #모든 문장의 공백을 제거 후 빈 문장은 제거한 새 리스트로 반환


def ensure_with_token_limit(text: str, target_tokenizer, limit: int = 512)-> str:
    #target_tokenizer 기준 토큰 수가 limit을 넘으면 문장 단위로 뒤어세부터 잘라 limit이하로 맞춤

    if not text:
        return text
    ids = target_tokenizer.encode(text, add_special_tokens=True)
    #add_special_tokens=True : 문장 시작과 끝에 각각 특수 토큰 추가

    if len(ids)<= limit:
        return text
    sentences = sentence(text)

    if not sentences:
        trimmed = target_tokenizer.decode(ids[:limit-1], skip_special_tokens=True)
        return trimmed
    acc = []
    for s in sentences:
        trial = ("".join(acc + [s]).strip())
        if len(target_tokenizer.encode(trial, add_special_tokens=True)) <= limit:
            acc.append(s)
        else:
            break

    if not acc:
        #문장 하나도 못담으면 하드컷
        trimmed = target_tokenizer.decode(ids[:limit-1], skip_special_tokens=True)
        return trimmed
    return " ".join(acc)


#기존 코드는 summarizer(i, ..)[0]에 빈 리스트가 들어와서 생기는 문제가 발생함
#따라서 예외처리를 하고 입력을 항상 리스트로 배치 호출로 바꿈 summarizer([part], ...)
def recursive_summarize( #gpt used
    text: str,
    summarizer,
    sum_tokenizer,
    max_input_tokens: int = 1024,
    stride: int = 128,
    gen_max_new_tokens: int = 256,
    gen_min_new_tokens: int = 32,
    gen_temperature: Optional[float] = None,) -> str:
    #매우 긴 기사는 chunk -> 각각요약 -> 결합 -> 필요 시 재요약
    if not isinstance(text, str) or not text.strip():
        return ""
    
    #문서를 토큰 기준으로 청크 분할
    chunks = split_into_chunks_by_tokens(text, tokenizer=sum_tokenizer, max_tokens=max_input_tokens, stride=stride)
    if not chunks:
        return text
    
    summaries = []
    for i in chunks:
        try:
            result = summarizer([i], gen_max_new_tokens = gen_max_new_tokens, min_new_tokens = gen_min_new_tokens)
            summary_text = result[0]["summary_text"].strip()
            summaries.append(summary_text)

        except Exception:
            summaries.append(i[:600])
            continue
        
    return "".join(summaries).strip()


#main
def main(args):
    #입력 로드
    df = pd.read_csv(args.input_csv)
    if "text" not in df.columns:
        raise ValueError("입력 csv에 'text'컬럼이 없습니다")
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU'if device == 0 else 'MPS'}")

    sum_tokenizer = AutoTokenizer.from_pretrained(args.summarizer_model, use_fast = True)
    sum_model = AutoModelForSeq2SeqLM.from_pretrained(args.summarizer_model)
    summarizer = pipeline("summarization", model = sum_model, tokenizer = sum_tokenizer, device = device)

    #최종 512 제한을 체크할 타깃 토크나이저 로드
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_tokenizer, use_fast = True)
    texts = df["text"].astype(str).tolist()
    results = []
    total = len(texts)

    for i, j in enumerate(texts):
        print(f"Processing {i+1}/{total}")
        summary = recursive_summarize(
            j,
            summarizer = summarizer,
            sum_tokenizer=sum_tokenizer,
            max_input_tokens=args.sum_max_input_tokens,
            stride = args.sum_stride,
            gen_max_new_tokens= args.sum_max_new_tokens,
            gen_min_new_tokens= args.sum_min_new_tokens,
            gen_temperature = None
        )

        #최종 512 토큰 제한 적용
        final_text = ensure_with_token_limit(
            summary,
            target_tokenizer= target_tokenizer,
            limit = args.target_max_tokens
        )
        results.append(final_text)

        if i % 50 == 0 or i == total:
            print(f"{i}/{total} done")

        #결과 저장
        df_out = pd.DataFrame({"text_512":results})
        for col in ("label", "id", "title"):
            if col in df.columns:
                df_out[col] = df[col]

        #칼럼 순서 정리: text_512, title, label, id, ...
        cols = ["text_512"]
        for i in ("title", "label", "id"):
            if i in df_out.columns:
                cols.append(i)
        cols += [c for c in df_out.columns if c not in cols]

        df_out.to_csv(args.output_csv, index = False, encoding = "utf-8-sig")
        print(f"saved {args.output_csv}(rows = {len(df_out)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type = str, default = "src/Fake_News_Detection_Data.csv")
    parser.add_argument("--output_csv", type = str , default= "src/Fake_News_Detection_Data_512.csv")
    parser.add_argument("--summarizer_model", type = str, default = "facebook/bart-large-cnn")
    parser.add_argument("--target_tokenizer", type = str, default = "klue/roberta-large")
    parser.add_argument("--sum_max_input_tokens", type = int, default = 1024)
    parser.add_argument("--sum_stride", type = int, default= 128)
    parser.add_argument("--sum_max_new_tokens", type= int, default= 256)
    parser.add_argument("--sum_min_new_tokens", type= int, default= 32)
    parser.add_argument("--target_max_tokens", type = int, default= 512)
    parser.add_argument("--cpu", action= "store_true", help = "강제로 CPU 사용")

    args = parser.parse_args()
    main(args)