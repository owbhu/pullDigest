import os, ijson, csv

BASE_DIR = "data/raw/kaggle-comments"
OUT_CSV  = "data/raw/kaggle-comments/pr_comments.csv"
MAX_RECS = 1000

def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    written = 0

    with open(OUT_CSV, "w", newline="", encoding="utf8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["repo", "number", "summary"])
        

        for root, dirs, files in os.walk(BASE_DIR):
            for fname in files:
                if not fname.endswith(".json"):
                    continue
                path = os.path.join(root, fname)
                try:
                    with open(path, "rb") as f:
                        for pr in ijson.items(f, "item"):
                            repo   = pr.get("repository") or pr.get("repo_name")
                            number = pr.get("number")     or pr.get("pr_number")
                            summary= pr.get("comment")    or pr.get("review_comment","")
                            if not repo or not number:
                                continue
                            writer.writerow([repo, number, summary])
                            written += 1
                            if written >= MAX_RECS:
                                print(f"Collected {written} records → {OUT_CSV}")
                                return
                except Exception as e:
                    print(f"error reading {path}: {e}")
    print(f"Finished with {written} records → {OUT_CSV}")

if __name__ == "__main__":
    main()
