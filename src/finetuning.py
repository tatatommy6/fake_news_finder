from transformers import BertForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd
import torch

# 전처리된 clickbait 데이터셋 불러오기
df1 = pd.read_csv("clickbait_dataset_final.csv")

# train/test 데이터를 stratify 기반으로 분할 (라벨 비율 유지)
# stratify란?: 기존 데이터에 나누는 것에 그치는게 아니라, 클래스 분포 비율까지 맞춰줌
train_df, eval_df = train_test_split(df1, test_size=0.2, stratify=df1["label"], random_state=42)

# HuggingFace Dataset 객체로 변환
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
eval_dataset = Dataset.from_pandas(eval_df.reset_index(drop=True))


model_name = "monologg/kobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)


model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# 전체 데이터셋에 토큰화 적용
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# 학습에 필요한 형식으로 dataset 구성 (input_ids, attention_mask, label만 사용)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

from transformers import TrainingArguments

# 학습 파라미터 설정 (gpt)
training_args = TrainingArguments(
    output_dir='./results',              # 체크포인트 및 로그 저장 폴더
    eval_strategy="epoch",              # 에폭마다 평가 실행(이거 좀 이상)
    num_train_epochs=3,                 # 전체 에폭 수
    per_device_train_batch_size=16,     # 학습 배치 크기
    per_device_eval_batch_size=32,      # 평가 배치 크기
    learning_rate=3e-5,                 # 학습률
    logging_strategy="epoch",           # 로그 출력 주기 (에폭마다)
    load_best_model_at_end=True,        # 가장 성능 좋은 모델 자동 복원
    save_strategy="epoch",              # 에폭마다 저장
    metric_for_best_model="accuracy",   # 최고 모델 기준 지표
    # keep_torch_compile = False        # (transformers 구버전일 경우 필요함)
    # trust_remote_code=True            # (동일 상황)
)

# 평가 지표 정의 함수 (정확도 계산)
def compute_metrics(eval_pred):
    logitcs, labels = eval_pred
    pred = torch.argmax(torch.tensor(logitcs), dim=1)
    acc = (pred == torch.tensor(labels)).float().mean()
    return {"accuracy": acc.item()}

# Trainer 객체 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# 모델 학습 시작
trainer.train()

# 학습된 모델 저장
trainer.save_model("./kobert_clickbait_model_final")
print("모델 훈련 및 저장 완료")