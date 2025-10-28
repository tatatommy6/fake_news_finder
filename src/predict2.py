import json, csv
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR   = Path("tatatommy6/finetuned-model-new")
INPUT_PATH  = Path("src/samples.jsonl")   # {"id":..., "label":"REAL|FAKE" 또는 정수, "text":"..."}
OUTPUT_PATH = Path("src/predictions.csv")
BATCH_SIZE  = 32
MAX_LENGTH  = 512


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_label_maps(model) -> Tuple[Dict[int, str], Dict[str, int]]:
    id2label = getattr(model.config, "id2label", None)
    label2id = getattr(model.config, "label2id", None)

    if not id2label:
        id2label = {0: "FAKE", 1: "REAL"}
    id2label = {int(k): str(v) for k, v in id2label.items()}

    if not label2id:
        label2id = {v: k for k, v in id2label.items()}
    else:
        label2id = {str(k): int(v) for k, v in label2id.items()}

    # 대/소문자, 공백, 숫자문자열까지 robust 매핑
    robust = {}
    for name, idx in label2id.items():
        robust[name] = idx
        robust[name.strip()] = idx
        robust[name.upper()] = idx
        robust[name.lower()] = idx
    label2id = robust
    return id2label, label2id


def normalize_true_label(raw_label: Any, id2label: Dict[int, str], label2id: Dict[str, int]) -> int:
    if isinstance(raw_label, int):
        if raw_label in id2label:
            return raw_label
        raise ValueError(f"Unknown integer label id: {raw_label}")
    s = str(raw_label).strip()
    if s.isdigit() and int(s) in id2label:
        return int(s)
    for key in (s, s.upper(), s.lower()):
        if key in label2id:
            return label2id[key]
    raise ValueError(f"Cannot map label '{raw_label}' to class id.")


def f1_from_confusion(conf: List[List[int]]) -> Tuple[float, float, float, Dict[int, float]]:
    """
    conf[true][pred]  (KxK)
    return: micro_f1, macro_f1, weighted_f1, per_class_f1
    """
    K = len(conf)
    tp = [conf[i][i] for i in range(K)]
    fp = [sum(conf[r][i] for r in range(K)) - conf[i][i] for i in range(K)]
    fn = [sum(conf[i]) - conf[i][i] for i in range(K)]
    support = [sum(conf[i]) for i in range(K)]
    total = sum(support) if K else 0

    # micro
    tp_sum, fp_sum, fn_sum = sum(tp), sum(fp), sum(fn)
    prec_micro = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0.0
    rec_micro  = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) else 0.0
    micro_f1 = (2 * prec_micro * rec_micro / (prec_micro + rec_micro)) if (prec_micro + rec_micro) else 0.0

    # per-class + macro/weighted
    per_class_f1, f1s, w_f1s = {}, [], []
    for i in range(K):
        p = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) else 0.0
        r = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        per_class_f1[i] = f1
        f1s.append(f1)
        w_f1s.append(f1 * (support[i] / total if total else 0.0))
    macro_f1 = sum(f1s) / K if K else 0.0
    weighted_f1 = sum(w_f1s) if total else 0.0
    return micro_f1, macro_f1, weighted_f1, per_class_f1


def main():
    device = pick_device()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()

    id2label, label2id = build_label_maps(model)
    num_labels = len(id2label)

    samples = list(read_jsonl(INPUT_PATH))
    texts   = [s["text"] for s in samples]

    y_true: List[int] = [normalize_true_label(s["label"], id2label, label2id) for s in samples]
    rows = []
    y_pred: List[int] = []

    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            enc = tok(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)
            confs, pred_ids = torch.max(probs, dim=-1)

            for j in range(len(batch)):
                pid = int(pred_ids[j])
                y_pred.append(pid)
                prob_cols = {f"prob_{id2label[k]}": float(probs[j, k]) for k in range(num_labels)}
                rows.append({
                    "id": samples[i+j].get("id", i+j),
                    "text": samples[i+j]["text"],
                    "true_label": id2label[y_true[i+j]],
                    "pred_label": id2label.get(pid, str(pid)),
                    "pred_confidence": round(float(confs[j]), 6),
                    **prob_cols
                })

    # CSV 저장
    if rows:
        with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # 메트릭 계산
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    accuracy = correct / len(y_true) if y_true else 0.0

    conf = [[0 for _ in range(num_labels)] for __ in range(num_labels)]
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1

    f1_micro, f1_macro, f1_weighted, per_class_f1 = f1_from_confusion(conf)

    per_class_f1_named = {id2label[i]: round(per_class_f1[i], 6) for i in range(num_labels)}
    conf_named = {id2label[i]: {id2label[j]: conf[i][j] for j in range(num_labels)} for i in range(num_labels)}

    metrics = {
        "num_samples": len(y_true),
        "accuracy": round(accuracy, 6),
        "f1_micro": round(f1_micro, 6),
        "f1_macro": round(f1_macro, 6),
        "f1_weighted": round(f1_weighted, 6),
        "per_class_f1": per_class_f1_named,
        "confusion_matrix": conf_named,
    }

    metrics_path = OUTPUT_PATH.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {OUTPUT_PATH} / {metrics_path} | 총 {len(rows)}건")
    print(f"acc={metrics['accuracy']}  f1_micro={metrics['f1_micro']}  f1_macro={metrics['f1_macro']}  f1_weighted={metrics['f1_weighted']}")


if __name__ == "__main__":
    main()
