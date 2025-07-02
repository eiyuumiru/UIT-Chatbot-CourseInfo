"""
map_dataset.py – create CSV mapping files
========================================
Convert **UITCourseInfo_dedup_qa.json** (422 context) to handy CSVs:

1. **course2ctx.csv**
   course_code, ctx_idx  ➜ tra nhanh chỉ mục context.
2. **qa_flat.csv**
   ctx_idx, course_code, question, answer_text  ➜ mỗi dòng 1 QA (đã khớp).

Run
---
$ python map_dataset.py  # assumes JSON file exists in cwd

Optional arguments:
    --in   path/to/json (default: UITCourseInfo_dedup_qa.json)
    --out  output dir    (default: ./maps)
"""

import json, argparse, csv
from pathlib import Path

####################################################################
# CLI args
####################################################################


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", "--input", dest="inp", default="UITCourseInfo_dedup_qa.json",
                   help="Input JSON {context,question,answer}")
    p.add_argument("--out", dest="out", default="maps",
                   help="Output directory for CSV files")
    return p.parse_args()

####################################################################
# main
####################################################################

def main():
    args = parse_args()
    inp = Path(args.inp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.load(open(inp, encoding="utf-8"))

    # 1) course2ctx.csv ------------------------------------------------
    (out_dir / "course2ctx.csv").write_text("course_code,ctx_idx\n" +
        "\n".join(f"{r['course_code']},{i}" for i,r in enumerate(rows)), encoding="utf-8")

    # 2) qa_flat.csv ---------------------------------------------------
    with (out_dir / "qa_flat.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ctx_idx", "course_code", "question", "answer_text"])
        for i, r in enumerate(rows):
            for q, a in zip(r["question"], r["answer"]):
                w.writerow([i, r["course_code"], q, a["text"][0]])

    print("✅ CSV written to", out_dir.resolve())


if __name__ == "__main__":
    main()
