# decode_val_anno.py
import json
import re
from pathlib import Path

INPUT  = Path("data/val_anno.json")
OUTPUT = Path("data/val_anno.decoded.json")

# \uXXXX 패턴이 실제로 남아있는지 확인용
_u_pat = re.compile(r'\\u[0-9a-fA-F]{4}')

def decode_unicode_once(s: str) -> str:
    # 문자열 안에 남아있는 \\uXXXX 이스케이프를 한 번만 안전하게 디코딩.
    if not _u_pat.search(s):
        return s
    # unicode_escape는 \n, \t 등도 디코딩하므로, 필요 이상 변환 방지용으로 한 번만 적용
    try:
        return s.encode("utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return s  # 문제 생기면 원문 유지

def walk(obj):
    # dict/list를 재귀적으로 순회하며 문자열만 선택적으로 디코딩.
    if isinstance(obj, dict):
        return {walk(k): walk(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [walk(x) for x in obj]
    if isinstance(obj, str):
        return decode_unicode_once(obj)
    return obj

def main():
    # 1. JSON 파싱 (선 디코딩 금지)
    raw = INPUT.read_text(encoding="utf-8-sig")
    # 일부 파일에 제어문자 등이 섞였을 경우 strict=False가 유용
    data = json.loads(raw, strict=False)

    # 2. 문자열만 선택 디코딩(이중 이스케이프 해소)
    data_decoded = walk(data)

    # 3. 예쁘게 저장 (한글을 그대로, 이스케이프 없이)
    with OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(data_decoded, f, ensure_ascii=False, indent=2)

    print(f"완료: {OUTPUT}")

if __name__ == "__main__":
    main()
