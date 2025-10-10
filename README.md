# 📰 Fake News Finder
가짜 뉴스 판독기 (RoBERTa + WEB Crawling)

---

Flask로 웹서버 구동 예정(공사중)  
Runpod를 이용해서 RTX 5090으로 학습 완료  
현재 웹사이트 다시 제작중 (Flask -> FastAPI)

---

## 설계 방식 (ver 1.0)

- 영문 기사 44,000개를 CSV 형태(`text`, `label`)로 구성.  
  `label`은 해당 기사가 가짜인지(1) 진짜인지(0)를 나타냄.

- 한국어 모델을 만들기 위해 [영→한 번역 모델](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-ko)을 사용.  
  RoBERTa 모델의 입력 제한(512 tokens)에 따라 전체 데이터셋의 20%를 **앞에서부터 512 토큰까지만 번역**함.

- 번역된 데이터셋을 [KLUE RoBERTa-large](https://huggingface.co/klue/roberta-large) 기반으로 파인튜닝.

- 최종적으로 파인튜닝된 모델은 [tatatommy6/fake_news_detect](https://huggingface.co/tatatommy6/fake_news_detect).

- 그러나 FastAPI 기반 데모 웹사이트 테스트 시 **정확도가 낮음**을 확인함.  
  (신뢰할 수 있는 기사도 거짓으로 분류하거나, 유명 사건을 가짜로 판단하는 등 문제 발생.)

---

## 문제 상황 정리

1. **정확도가 낮았던 이유**
   - 데이터 전처리 설계 오류로 판단.
   - 번역 전 자르기 vs 번역 후 자르기 기준이 불명확.
   - 기사 핵심이 후반부에 존재할 경우 정보 손실이 심함.
   - 번역 과정에서 고유명사·부정 표현 왜곡.
   - 가짜 뉴스 탐지는 문체보다는 **팩트 검증(Claim Verification)** 성격이 강함.

2. **해결 방법**
   - 긴 기사 텍스트를 **영문 요약 모델**로 압축.
   - 요약모델 입력 한계(1024 tokens)에 따라 **청크 분할 → 각 청크 요약 → 병합**.
   - 병합 결과가 512 tokens 초과 시 **재요약** 진행.
   - 최종 결과를 **512토큰 이하 문장 단위로 정제**하여 CSV 저장.
   - 이 데이터셋을 기반으로 학습 구조를 다시 설계.

---

## 개선된 설계 방식 (ver 2.0)

- **핵심 목표:**  
  RoBERTa의 입력 제한(512 tokens)을 지키면서 기사 핵심 정보를 최대한 보존.

- **데이터 전처리 단계:**
  1. 기사 청크 분할 (1024 tokens 단위)  
  2. 각 청크 요약 (BART / Pegasus 등 사용)  
  3. 요약문 병합 및 재요약  
  4. 문장 단위 트리밍으로 512 tokens 이하로 맞춤  
  5. 결과를 `text_512`, `label` 형식 CSV로 저장  

- **결과:**  
  기존보다 의미 손실이 적고, 한 문장에 정보가 밀도 높게 담긴 데이터 확보.

---

## 추가 설계 (ver 2.1)  
> 요약된 영어 CSV → 한국어 번역 → 한국어 RoBERTa 학습 및 웹 서비스 단계

### 1. EN→KO 번역 단계
- 요약 후 생성된 `text_512`(영문) 데이터를  
  [Helsinki-NLP/opus-mt-tc-big-en-ko](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-ko)  
  모델을 사용해 한국어로 번역.  
- 번역 결과를 `text_512_ko`, `label` 형식의 CSV로 저장.  
- 이 과정을 통해 **짧고 의미 있는 문장만 번역**되므로 번역 품질이 향상됨.

### 2. 한국어 모델 파인튜닝 단계
- 번역된 한국어 데이터셋(`text_512_ko`, `label`)을  
  [KLUE RoBERTa](https://huggingface.co/klue/roberta-base) 기반으로 파인튜닝.  
- 입력 길이는 512 토큰으로 고정하여 안정적인 학습을 수행.  

### 3. FastAPI 기반 웹 서비스
- 학습된 모델을 FastAPI로 배포하여 구현 예정
- 사용자는 텍스트를 입력하면 모델이 확률(`prob_fake`)과 결과(`label`)을 반환.

### 4. 2차 검증 단계 (웹 크롤링)
- 모델이 1차로 “가짜” 판별한 기사를 대상으로  
  **검색엔진 크롤링 / 뉴스 API**를 통해 동일 사건 교차검증 수행.  
- **여러 신뢰 매체에서 동일 사건 존재 시 → 진짜**  
  **단일/비신뢰 출처만 존재 시 → 가짜 가능성 높음**  

---

## 결과 요약

| 구분 | ver 1.0 | ver 2.0~2.1 |
|------|----------|--------------|
| 데이터 처리 방식 | 번역 후 앞부분 자르기 | 요약 → 번역 → 학습 |
| 토큰 효율성 | 낮음 (불필요 문장 다수) | 높음 (핵심 문장 중심) |
| 번역 품질 | 긴 문장 번역으로 오류 다수 | 짧은 요약문 번역으로 품질 향상 |
| 모델 언어 | 영어 일부 혼용 | 완전한 한국어 모델 |
| 확장성 | 없음 | 웹서버 + 2차 검증 가능 |
| 기대 효과 | 낮은 정확도 | **정확도 및 신뢰도 상승** |

---

## 향후 계획

- [x] 영어 기사 요약 파이프라인 완성  
- [x] EN→KO 번역 및 CSV 변환 완료  
- [ ] KLUE RoBERTa 파인튜닝 및 검증  (ver 1.0 완료 2.0 & 2.1 설계중)
- [ ] FastAPI 서비스 배포  
- [ ] 크롤링 기반 2차 검증 알고리즘 구현  
- [ ] 전체 파이프라인 자동화 (CLI or notebook 기반)

---

## 참고 모델

- 요약: [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)
- 번역: [Helsinki-NLP/opus-mt-tc-big-en-ko](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-ko)  
- 한국어 분류: [klue/roberta-large](https://huggingface.co/klue/roberta-large)

---

## 요약

> ver 2.0~2.1은 **요약 기반 전처리 → 번역 → 한국어 RoBERTa 학습 → FastAPI 서비스 → 웹 크롤링 검증**까지  
> 전체 파이프라인을 완성한 버전입니다.  
>  
> 이를 통해 입력 제한(512 tokens)을 지키면서 정보 손실을 최소화하고,  
> 한국어 환경에서도 높은 신뢰도의 가짜뉴스 탐지 모델을 구현할 수 있게 되었습니다.
