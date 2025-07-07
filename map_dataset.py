import json, argparse, csv
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="UITCourseInfo_dedup_qa.json",
                   help="Input JSON file from prepare_dataset.py")
    p.add_argument("--out", dest="out", default="course_info.csv",
                   help="Output CSV path")
    return p.parse_args()

def main():
    args = parse_args()
    inp = Path(args.inp)
    out_csv = Path(args.out)

    rows = json.load(open(inp, encoding="utf-8"))

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["course_id", "course_code", "context", "question", "answer"])

        for idx, r in enumerate(rows):
            ctx   = r["context"].replace("\n", " ").strip()
            code  = r.get("course_code", "")
            for q, a in zip(r["question"], r["answer"]):
                w.writerow([idx, code, ctx, q, a["text"][0]])

    print("CSV saved to", out_csv.resolve())


if __name__ == "__main__":
    main()