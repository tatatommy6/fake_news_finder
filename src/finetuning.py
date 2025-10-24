import os
import random
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, EarlyStoppingCallback
from transformers import Trainer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# csv 로드
df = pd.read_csv("/src/Fake_News_Detection_Data_512.csv") # 임시

if df['label'].dtype == object:
    mapping = {k : i for i, k in enumerate(sorted(df['label'].unique()))}
    df['label'] = df['label'].map(mapping)

train_df, val_df = train_test_split(df, test_size = 0.2, random_state = SEED, stratify = df['label'])
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(val_df)

# 토크나이저 및 모델 로드
MODEL_NAME = "klue/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels = 2)

def tokenizer_function(batch):
    return tokenizer(batch['text'], truncation = True, padding = False, max_length = 512)

train_dataset = train_dataset.map(tokenizer_function, batched = True, desc = "Tokenizing train")
valid_dataset = valid_dataset.map(tokenizer_function, batched = True, desc = "Tokenizing train")

# Trainer 입력 컬럼만 남기기
cols = ['input_ids', 'attention_mask', 'label']
train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in cols])
valid_dataset = valid_dataset.remove_columns([c for c in valid_dataset.column_names if c not in cols])

data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

# 매트릭스 함수 정의
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis = -1)
    acc = accuracy_score(labels, preds)
    prec, recall, f1 = precision_recall_fscore_support(labels, preds, average = 'binary', zero_division = 0)
    return {'accuracy': acc, 'precision': prec, 'recall': recall, 'f1': f1}

# 학습 파라미터 세팅
# klue/roberta-large 는 vram 부담이 크므로 배치를 줄이고 필요 시 gradient_accumulation 으로 보완
args = TrainingArguments(
    output_dir = "/src/finetuned_model",
    evaluation_strategy = "steps",
    logging_strategy = "steps",
    save_strategy = "steps",
    eval_steps = 100,
    save_steps = 500,
    logging_steps = 100,
    per_device_eval_batch_size = 16,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 1,
    num_train_epochs = 3,
    learning_rate = 2e-5,
    warmup_ratio = 0.1,
    weight_decay = 0.01,
    load_best_model_at_end = True,
    metric_for_best_model = "f1",
    greater_is_better = True,
    fp16 = torch.cuda.is_available(),
    save_total_limit = 2,
    seed = SEED,
    dataloader_num_workers = 2,
    report_to = "none"
)

callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)]

trainer = Trainer(
    model = model,
    args = args,
    train_dataset = train_dataset,
    eval_dataset = valid_dataset,
    compute_metrics = compute_metrics,
    tokenizer = tokenizer,
    data_collator = data_collator,
    callbacks = callbacks
)

#학습 실행
trainer.train()
#검증 점수 출력
print(trainer.evaluate())

#추론 예시
test_samples = [
    "정부가 조작한 비밀 문서가 발견되었다는 글이 확산되고 있습니다.",
    "오늘 발표된 통계청 자료에 따르면 실업률은 작년 대비 0.2%p 하락했습니다."
]

enc = tokenizer(test_samples, return_tensors="pt", padding=True, truncation=True, max_length=512)
model.eval()
with torch.no_grad():
    out = model(**{k : v.to(model.device) for k,v in enc.items()})
probs = out.logits.softmax(-1).cpu().numpy()
preds = probs.argmax(-1)
print(list(zip(test_samples, preds, probs)))