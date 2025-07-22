import os
import json
import pandas as pd
from tqdm import tqdm

# clickbait, nonclickbait 폴더 경로 설정
base_dirs = {
    "clickbait": "/Users/kimminkyeol/Programming/dataset/fakeorrealdata/datas/clickbait",
    "nonclickbait": "/Users/kimminkyeol/Programming/dataset/fakeorrealdata/datas/noneclickbait"
}

def collect_labeled_titles(base_dirs, dirname:str):
    data = []
    dirname = dirname.lower()  # 소문자로 변환하여 비교
    for label_name, base_dir in base_dirs.items():
        label_value = 1 if label_name == dirname else 0

        for root, _, files in os.walk(base_dir): #os.walk(): 특정 경로 내에 존재하는 디렉토리와 파일 리스트 뿐만 아니라, 모둔 하위 디렉토리 까지 검색
            for filename in sorted(files):  #sorted(): 주어진 iterable 객체를 정렬하여 새로운 리스트로 반환하는 내장 함수
                if not filename.endswith("_L.json"): #이걸로 끝나지 않으면
                    continue #skip
                path = os.path.join(root, filename)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        raw = json.load(f)
                        labeled = raw.get("labeledDataInfo", {}) #labeledDataInfo 테그를 찾음
                        if "newTitle" not in labeled: #newtitle 테그가 없으면
                            continue  # skip
                        title = labeled["newTitle"].strip()
                        if not title:
                            continue  # 빈 문자열도 skip
                        data.append({"text": title, "label": label_value}) #데이터를 딕셔너리 형태로 저장
                except Exception as e: # 예외처리
                    print(f"exception: {path} - {e}")
                    continue
    return pd.DataFrame(data)

# csv로 저장
df = collect_labeled_titles(base_dirs, "clickbait") #파일명 바꾸는 곳
df.to_csv("notclickbait_dataset_final.csv", index=False, encoding='utf-8-sig')