import os
import sqlite3
import pandas as pd

def build_small_dataset(n=1000, out_csv="data/small_prs.csv"):
    """
    Build a smaller dataset of PR diffs with gold summaries (PR body).
    Samples up to n entries where diff length < 10k and body is non-empty.
    Writes CSV with columns: repo, number, diff, summary.
    """

    conn = sqlite3.connect("data/prs.db")
    df_prs = pd.read_sql("SELECT repo, number, diff, body FROM prs", conn)
    conn.close()


    df_prs = df_prs[df_prs["body"].str.strip().astype(bool)]


    df_prs = df_prs[df_prs["diff"].str.len() < 10000]

    total = len(df_prs)
    if total == 0:
        raise RuntimeError("No PRs found after filtering—check your database and filters.")


    sample_size = min(n, total)
    if sample_size < n:
        print(f"Only {total} PRs available; sampling all of them.")
    small = df_prs.sample(sample_size, random_state=42)


    small = small.rename(columns={"body": "summary"})


    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    small[["repo", "number", "diff", "summary"]].to_csv(out_csv, index=False)
    print(f"Written {len(small)} PRs to {out_csv}")

if __name__ == "__main__":
    build_small_dataset()
