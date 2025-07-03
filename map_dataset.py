from huggingface_hub import hf_hub_download
import pandas as pd
import argparse, json, re, os, tempfile

PAT_CODE = re.compile(r"\b[A-Z]{2,}\d{3}\b")

def extract_code(text: str | None):
    m = PAT_CODE.search(text or "")
    return m.group(0) if m else None

def build_map(df: pd.DataFrame):
    codes = (
        df[["course_code"]]
        .drop_duplicates()
        .sort_values("course_code")
        .reset_index(drop=True)
    )
    codes["course_id"] = codes.index + 1
    return dict(zip(codes.course_code, codes.course_id))

def main(repo_id, json_name, map_file, clean_file):
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=json_name,
        repo_type="dataset",
        local_dir=tempfile.mkdtemp(),
    )
    with open(local_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for split in ("train", "validation", "test"):
        for ex in raw.get(split, []):
            ex["split"] = split
            records.append(ex)
    df = pd.DataFrame(records)

    df["course_code"] = df["context"].apply(extract_code)
    df = df.dropna(subset=["course_code"])

    code2id = build_map(df)

    df["course_id"] = df["course_code"].map(code2id)
    df.to_csv(clean_file, index=False, encoding="utf-8")
    with open(map_file, "w", encoding="utf-8") as f:
        json.dump(code2id, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="PhucDanh/UIT-CourseInfo")
    ap.add_argument("--json-name", default="UITCourseInfo.json")
    ap.add_argument("--map-file", default="course_map.json")
    ap.add_argument("--clean-file", default="course_clean.csv")
    args = ap.parse_args()
    main(**vars(args))