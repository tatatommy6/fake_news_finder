import os
import zipfile
from tqdm import tqdm

def unzip_and_sort(zip_dir, output_dir):
    os.makedirs(os.path.join(output_dir, "clickbait"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "nonclickbait"), exist_ok=True)

    for file in tqdm(os.listdir(zip_dir)):
        if not file.endswith(".zip"):
            continue

        zip_path = os.path.join(zip_dir, file)

        if "Clickbait" in file and "NonClickbait" not in file:
            label_type = "clickbait"
        elif "NonClickbait" in file:
            label_type = "nonclickbait"
        else:
            print(f"❗ Unrecognized file: {file}")
            continue

        folder_name = os.path.splitext(file)[0]
        extract_path = os.path.join(output_dir, label_type, folder_name)
        os.makedirs(extract_path, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    print("✅ 압축 해제 및 정리 완료.")

# 경로 지정
zip_dir = "/Users/kimminkyeol/Programming/dataset/fakeorrealdata/datas/Training/02.labeld"
output_dir = "/Users/kimminkyeol/Programming/dataset/fakeorrealdata/datas/"
unzip_and_sort(zip_dir, output_dir)
