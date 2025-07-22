import os
import json
import pandas as pd
from tqdm import tqdm

# clickbait, nonclickbait 폴더 경로 설정
base_dirs = {
    "clickbait": "/Users/kimminkyeol/Programming/dataset/fakeorrealdata/datas/clickbait",
    "nonclickbait": "/Users/kimminkyeol/Programming/dataset/fakeorrealdata/datas/noneclickbait"
}

# 모든 JSON을 탐색하며 newTitle과 clickbaitClass 추출
def collect_labeled_titles(base_dirs):
    data = []
    for label_name, base_dir in base_dirs.items():
        label_value = 1 if label_name == "clickbait" else 0

        for root, _, files in os.walk(base_dir):
            for fname in files:
                if not fname.endswith(".json") or not fname.endswith("_L.json"):
                    continue
                path = os.path.join(root, fname)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        raw = json.load(f)
                        title = raw["labeledDataInfo"]["newTitle"].strip()
                        data.append({"text": title, "label": label_value})
                except Exception as e:
                    print(f"⚠️ 오류 발생: {path} - {e}")
                    continue
    return pd.DataFrame(data)

# 데이터 수집 및 저장
df = collect_labeled_titles(base_dirs)
df.to_csv("clickbait_dataset_final.csv", index=False, encoding='utf-8-sig')