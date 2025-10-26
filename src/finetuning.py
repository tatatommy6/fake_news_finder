import os
import random
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,
    DataCollatorWithPadding
)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# 1) 데이터 로드 & 라벨 매핑
df = pd.read_csv("tras_final.csv")

if df['labels'].dtype == object:
    # 문자열 라벨일 때 안정적인 매핑 고정(사전순이 아니라 데이터에 의존하지 않도록)
    unique = sorted(df['labels'].unique().tolist())
    mapping = {lbl:i for i, lbl in enumerate(unique)}
    df['labels'] = df['labels'].map(mapping)
else:
    mapping = None  # 이미 숫자 라벨

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df['labels']
)
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
valid_dataset = Dataset.from_pandas(val_df.reset_index(drop=True))

# 2) 토크나이저/모델
MODEL_NAME = "klue/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
num_labels = int(df['labels'].nunique())
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

# 라벨 이름(가능하면 원래 문자열 보존)
if mapping is not None:
    id2label = {i: lbl for lbl, i in mapping.items()}
    label2id = {lbl: i for i, lbl in id2label.items()}
else:
    id2label = {i: str(i) for i in range(num_labels)}
    label2id = {v: k for k, v in id2label.items()}

model.config.id2label = id2label
model.config.label2id = label2id

def tokenizer_function(batch):
    return tokenizer(batch['text'], truncation=True, padding=False, max_length=512)

train_dataset = train_dataset.map(tokenizer_function, batched=True, desc="Tokenizing train")
valid_dataset = valid_dataset.map(tokenizer_function, batched=True, desc="Tokenizing valid")

# 3) Trainer 입력 컬럼 정리
cols = ['input_ids', 'attention_mask', 'labels']
train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in cols])
valid_dataset = valid_dataset.remove_columns([c for c in valid_dataset.column_names if c not in cols])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 4) 매트릭스: macro F1로 클래스 불균형/양성라벨 혼선 방지
def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    return {'accuracy': acc, 'precision_macro': prec, 'recall_macro': recall, 'f1_macro': f1}

# 5) 학습 파라미터: 에폭 단위 평가/저장 & 중간종료 제거
args = TrainingArguments(
    output_dir="finetuned_model_new",
    overwrite_output_dir=True,

    num_train_epochs=3,
    max_steps=-1,  # 에폭 기준
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    logging_first_step=True,

    save_total_limit=2,
    load_best_model_at_end=False,     # 에폭 끝까지 달리기
    metric_for_best_model="f1_macro",
    greater_is_better=True,

    # 성능/안정성
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,    # VRAM 부족하면 4로 낮추고, 여기 2로 올려 효과배치 유지
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=0,         # 워커로 인해 드문 행업 방지(필요시 2로 올리세요)
    report_to="none",
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 6) 학습 (체크포인트 재개 금지로 반쪽 에폭 종료 방지)
trainer.train(resume_from_checkpoint=False)

# 7) 검증 점수
print(trainer.evaluate())

# 8) 추론 예시
test_samples = [
    "정부가 조작한 비밀 문서가 발견되었다는 글이 확산되고 있습니다.",
    "오늘 발표된 통계청 자료에 따르면 실업률은 작년 대비 0.2%p 하락했습니다."
]
enc = tokenizer(test_samples, return_tensors="pt", padding=True, truncation=True, max_length=512)
model.eval()
with torch.no_grad():
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model(**enc)
probs = out.logits.softmax(-1).cpu().numpy()
preds = probs.argmax(-1)
print(list(zip(test_samples, preds, probs)))