import pandas as pd
import sys

FULL_PATH = "src/trans_kor.csv"  # 컬럼: text,labels, translated_text
FIX_PATH = "src/retrans_fixed.csv"  # 컬럼: text, labels, translated_text, len_ratio_ok, has_bad, gen_len
OUT_PATH = "src/tras_final.csv"  # 최종 컬럼: text, labels(중복 제거)
TARGET_COL = "translated_text"


def normalize(s: pd.Series) -> pd.Series:
    # 텍스트 컬럼의 공백 정리 및 앞뒤 공백 제거
    return (
        s.astype(str)
        .str.replace(r"\s+"," ", regex = True)
        .str.strip()
    )


def main():
    # 1. 로드 & 필수 컬럼 체크
    full = pd.read_csv(FULL_PATH)
    fix = pd.read_csv(FIX_PATH)

    need_full = {"text", "labels", TARGET_COL}
    need_fix = {"text", TARGET_COL}

    if not need_full.issubset(full.columns):
        sys.exit(f"error: full file missing colnms: {need_full - set(full.columns)}")
    if not need_fix.issubset(fix.columns):
        sys.exit(f"error: fixed file missing colnms: {need_fix - set(fix.columns)}")

    # 2. 매칭 키: text(정규화)
    full["_k"] = normalize(full["text"])
    fix["_k"] = normalize(fix["text"])

    fix_sub = ( #trans_text 컬럼만 추출한 재번역 텍스트 레이블
        fix[["_k", TARGET_COL]]
            .drop_duplicates(subset = ["_k"], keep="first")
            .rename(columns={TARGET_COL: f"{TARGET_COL}_new"})
    )

    # 3. left-join 후 필요한 컬럼만 선택
    merged = pd.merge( # 여기서 재번역된 텍스트를 전체 데이터셋에 덧붙임
        full,
        fix_sub,
        on = "_k",
        how = "left",
        validate = "m:1" # 전체 파일은 문장 중복이 있을 수도 있지만, 재번역 파일에서는 같은 문장이 중복되면 안 된다.”라는 전제에 맞춰 "m:1"을 사용
                        # 잘못된 중복이 생기면 즉시 에러를 띄워주는 데이터 무결성 체크 장치
    )

    #병합 후 새로 붙은 translated_text_new로 기존 번역(translated_text)을 덮어쓰기
    merged[TARGET_COL] = merged[f"{TARGET_COL}_new"].combine_first(merged[TARGET_COL]) 
    # 즉, translated_text_new 값이 translated_text에 존재하면 그것을 사용하고, 그렇지 않으면 기존의 translated_text 값을 유지

    # 4. 최종 출력 구성: translated_text -> text, 그리고 labels
    out = (
        merged[[TARGET_COL, "labels"]]
        .rename(columns={TARGET_COL: "text"})
    )

    # 5. 정리: 공백·결측 제거 및 (text, labels) 중복 제거
    out["text"] = normalize(out["text"])
    out["labels"] = out["labels"].astype(str).str.strip()

    out = out[(out["text"].notna()) & (out["text"] != "")]
    out = out[(out["labels"].notna()) & (out["labels"] != "")]
    out = out.drop_duplicates(subset = ["text", "labels"], keep = "first")

    # 6. 저장
    out.to_csv(OUT_PATH, index = False, encoding = "utf-8")
    print(f"done. saved to {OUT_PATH}")

if __name__ == "__main__":
    main()
