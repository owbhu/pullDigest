# src/fetch_prs.py

import os
import sys
import sqlite3
import requests
from concurrent.futures import ThreadPoolExecutor
from github import Github
from src.config import GITHUB_TOKEN, REPO_LIST, DB_PATH

# GraphQL query with pagination
_GQL = """
query($owner:String!,$name:String!,$first:Int!,$after:String) {
  repository(owner:$owner, name:$name) {
    pullRequests(first:$first, after:$after,
                 orderBy:{field:UPDATED_AT,direction:DESC}) {
      pageInfo { endCursor, hasNextPage }
      nodes {
        number
        title
        body
      }
    }
  }
}
"""

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
      CREATE TABLE IF NOT EXISTS prs (
        repo   TEXT,
        number INTEGER,
        title  TEXT,
        body   TEXT,
        diff   TEXT,
        PRIMARY KEY (repo, number)
      )""")
    return conn

def fetch_prs(limit=100):
    if not GITHUB_TOKEN:
        raise RuntimeError("Please set GITHUB_TOKEN in your environment")

    gh      = Github(GITHUB_TOKEN)
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    conn    = init_db()
    cursor  = conn.cursor()

    for full in REPO_LIST:
        owner, name = full.split("/")
        print(f"▶ Fetching up to {limit} PRs for {full} via GraphQL…")

        fetched = []
        cursor_str = None

        # Loop until we have `limit` or no more pages
        while len(fetched) < limit:
            batch_size = min(limit - len(fetched), 100)
            variables  = {
                "owner": owner,
                "name":  name,
                "first": batch_size,
                "after": cursor_str
            }

            resp = requests.post(
                "https://api.github.com/graphql",
                json={"query": _GQL, "variables": variables},
                headers=headers
            )
            j = resp.json()
            if resp.status_code != 200 or "errors" in j:
                print("GraphQL error:", resp.status_code,
                      j.get("message", j.get("errors")))
                sys.exit(1)

            pr_data   = j["data"]["repository"]["pullRequests"]
            nodes     = pr_data["nodes"]
            page_info = pr_data["pageInfo"]

            print(f"  • Retrieved {len(nodes)} PRs (total {len(fetched)+len(nodes)})")
            fetched.extend(nodes)

            if not page_info["hasNextPage"]:
                break
            cursor_str = page_info["endCursor"]

        total = len(fetched)
        print(f"  ✓ Fetched metadata for {total} PRs in {full}")

        # Download diffs in parallel
        def fetch_diff(node):
            pr_num = node["number"]
            diff_url = f"https://api.github.com/repos/{full}/pulls/{pr_num}.diff"
            try:
                diff = requests.get(diff_url, headers=headers).text
                return (
                    full,
                    pr_num,
                    node.get("title") or "",
                    node.get("body")  or "",
                    diff
                )
            except Exception as e:
                print(f"Error fetching diff for {full}#{pr_num}: {e}")
                return None

        print("  • Downloading diffs in parallel (8 workers)…")
        with ThreadPoolExecutor(max_workers=8) as exe:
            for i, result in enumerate(exe.map(fetch_diff, fetched), start=1):
                if result:
                    cursor.execute(
                        "INSERT OR IGNORE INTO prs (repo,number,title,body,diff) VALUES (?,?,?,?,?)",
                        result
                    )
                if i % 50 == 0 or i == total:
                    print(f"    • {i}/{total} diffs processed")

        conn.commit()

    conn.close()
    print(f"All PRs fetched into {DB_PATH}")

if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    fetch_prs(limit=lim)
