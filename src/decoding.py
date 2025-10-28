import json

in_path = "data/val_anno.json"

out_path = "data/val_data.json"

with open(in_path, "r", encoding = "utf-8-sig") as f:
    raw = f.read()

decode = raw.encode("utf-8").decode("unicode_escape") 

try:
    data = json.loads(decode)
    with open(out_path, "w", encoding="uth-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Decoding successful, data written to", out_path)
except json.JSONDecodeError as e:
    print("Failed to decode JSON:", e)
