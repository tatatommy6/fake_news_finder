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
import padnas as pd
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
    if not text:
        return ""
    
    #1. 긴 문서를 청크로 나눠 각각 요약
    chunks = split_into_chunks_by_tokens(text, tokenizer = sum_tokenizer, max_toknes = max_input_tokens, stride = stride)

    if len(chunks) == 1:
        parts = [chunks[0]]
    else:
        parts = chunks
    
    summaries = []
    for i in parts:
        gen_kwargs = {
            "max_new_tokens" : gen_max_new_tokens,
            "min_new_tokens" : gen_min_new_tokens,
            "do_sample" : gen_temperature is not None,
        }
        if gen_temperature is not None:
            gen_kwargs["temperature"] = gen_temperature

        out = summarizer(i **gen_kwargs)[0]["summary_text"]
        summaries.append(out.strip())

    merged = "".join(summaries).strip()

    #2 결합 결과가 요약 모델 입력 한계를 넘으면 재요약
    if len(sum_tokenizer.encode(merged, add_special_tokens=False)) > max_input_tokens:
        out = summarizer(merged, max_new_tokens = gen_max_new_tokens, min_new_tokens = gen_min_new_tokens, do_sample = False)[0]["summary_text"].strip()
        return out
    return merged