import json, csv
from pathlib import Path
from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = Path("tatatommy6/finetuned-model-new")
INPUT_PATH = Path("samples.jsonl")
OUTPUT_PATH = Path("predictions.csv")
BATCH_SIZE = 32
MAX_LENGTH = 512

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    return torch.device("cpu")

def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

device = pick_device()
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()

# 라벨 매핑 확인(없으면 기본 추정)
id2label: Dict[int, str] = getattr(model.config, "id2label", None) or {0:"FAKE", 1:"REAL"}
num_labels = len(id2label)

samples = list(read_jsonl(INPUT_PATH))
texts = [s["text"] for s in samples]

rows = []
with torch.no_grad():
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        enc = tok(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)
        confs, pred_ids = torch.max(probs, dim=-1)

        for j in range(len(batch)):
            pid = int(pred_ids[j])
            label = id2label.get(pid, str(pid))
            conf  = float(confs[j])
            prob_cols = { f"prob_{id2label[k]}": float(probs[j,k]) for k in range(num_labels) }
            rows.append({
                "id": samples[i+j]["id"],
                "text": samples[i+j]["text"],
                "pred_label": label,
                "pred_confidence": round(conf, 6),
                **prob_cols
            })

# 저장
if rows:
    import csv
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

print(f"✅ 저장 완료: {OUTPUT_PATH} (총 {len(rows)}건) | device={device}")
