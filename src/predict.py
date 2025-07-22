from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import json

# Load model directly

tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")
model = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-FinBert-SC")

classifier = pipeline(
    "text-classification",
    model="snunlp/KR-FinBert-SC"
)

# 2. JSON 불러오기
with open("result.json", "r", encoding="utf-8") as f:
    news_list = json.load(f)

# 3. 기사 하나씩 분류
for article in news_list:
    title = article  # 이미 문자열
    result = classifier(title)[0]
    label = result['label']
    score = round(result['score'] * 100, 2)
    print(f"[{title}] → 예측: {label} ({score}%)")