# pullDigest

Automated pipeline for abstractive summarization of GitHub pull requests using transformer models.

Author: Owen Hughes
Contact: obh@uoregon.edu

## Quick Start

### 1. Clone & enter the repo
```bash
git clone https://github.com/owbhu/pullDigest.git
cd pullDigest
````

### 2. Set up your environment

```bash
conda create -n pulldigest python=3.11 -y
conda activate pulldigest
conda install -c conda-forge pygithub requests tiktoken transformers datasets evaluate rouge-score absl-py nltk -y
export TOKENIZERS_PARALLELISM=false
# Create a .env in the repo root with:
# GITHUB_TOKEN=ghp_XXXXXXXXXXXXXXXXXXXX
```

### 3. Fetch GitHub PRs

```bash
python -m src.fetch_prs 500
```

### 4. Build & sample dataset

```bash
python -m src.sample_data
python - << 'PY'
import pandas as pd
df    = pd.read_csv("data/small_prs.csv")
train = df.sample(frac=0.8, random_state=42)
hold  = df.drop(train.index)
dev   = hold.sample(frac=0.5, random_state=42)
test  = hold.drop(dev.index)
train.to_csv("data/train.csv", index=False)
dev.to_csv("data/dev.csv",   index=False)
test.to_csv("data/test.csv", index=False)
print("Splits:", len(train), len(dev), len(test))
PY
```

### 5. Evaluate models (ROUGE-L)

```bash
python - << 'PY'
import pandas as pd
from evaluate import load
from src.summarizer import summarize_chunk
from transformers import pipeline

def eval_model(model_id, split):
    df = pd.read_csv(f"data/{split}.csv")
    rouge = load("rouge")
    if model_id.startswith("facebook"):
        summ = pipeline("summarization", model=model_id, device=0)
        preds = [summ(d, max_length=75, min_length=15, do_sample=False)[0]["summary_text"]
                 for d in df["diff"].astype(str)]
    else:
        preds = [summarize_chunk(d) for d in df["diff"].astype(str)]
    refs = [[s] for s in df["summary"].astype(str)]
    score = rouge.compute(predictions=preds, references=refs)["rougeL"]
    print(f"{model_id} {split} ROUGE-L: {score:.4f}")

for mid in ["google/flan-t5-small","facebook/bart-large-cnn"]:
    for split in ["dev","test"]:
        eval_model(mid, split)
PY
```

### 6. (Optional) Fine-tune FLAN-T5-small

```bash
python -m src/finetune_flant5
```

## Repo Structure

```
pullDigest/
├── data/
│   ├── prs.db
│   ├── small_prs.csv
│   ├── train.csv
│   ├── dev.csv
│   └── test.csv
├── src/
│   ├── config.py
│   ├── fetch_prs.py
│   ├── sample_data.py
│   ├── chunker.py
│   ├── summarizer.py
│   └── finetune_flant5.py
├── .env
└── README.md
```
