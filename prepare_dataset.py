import json, requests, unicodedata, re
from pathlib import Path
import pandas as pd

HF_URL = (
    "https://huggingface.co/datasets/PhucDanh/UIT-CourseInfo/"
    "resolve/main/UITCourseInfo.json"
)

def download_json(url: str) -> dict:
    return requests.get(url, timeout=60).json()


def normalize(txt: str) -> str:
    txt = unicodedata.normalize("NFKC", txt.strip().lower())
    return re.sub(r"\s+", " ", txt)

def dedup_qa(pairs):
    """pairs = list[(question, answer_dict)]  -> dedup theo answer_start"""
    seen, keep_q, keep_a = set(), [], []
    for q, a in pairs:
        start = a.get("answer_start", [None])[0]
        if start in seen:
            continue
        seen.add(start)
        keep_q.append(q)
        keep_a.append(a)
    return keep_q, keep_a

def main():
    rows = sum(download_json(HF_URL).values(), [])
    df = pd.DataFrame(rows)

    df["course_code"] = df["context"].str.extract(r"\b([A-Z]{2,3}\d{3,4})\b", expand=False)
    df["norm_ctx"]   = df["context"].map(normalize)

    grouped = (
        df.groupby("norm_ctx", sort=False)
          .agg({
              "context": "first",
              "course_code": "first",
              "question":  list,
              "answer":    list,
          })
          .reset_index(drop=True)
    )

    new_q, new_a = [], []
    for qs, ans in zip(grouped["question"], grouped["answer"]):
        q_keep, a_keep = dedup_qa(list(zip(qs, ans)))
        new_q.append(q_keep)
        new_a.append(a_keep)
    grouped["question"], grouped["answer"] = new_q, new_a

    out = Path("UITCourseInfo_dedup_qa.json")
    json_str = grouped.to_json(orient="records", force_ascii=False, indent=2)
    out.write_text(json_str, encoding="utf-8")
    print(f"Saved {len(grouped)} context to {out}")


if __name__ == "__main__":
    main()
