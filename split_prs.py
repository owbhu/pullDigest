# split small_prs.csv into 80/10/10
import pandas as pd

df = pd.read_csv("data/small_prs.csv")
train = df.sample(frac=0.8, random_state=42)
hold  = df.drop(train.index)
dev   = hold.sample(frac=0.5, random_state=42)
test  = hold.drop(dev.index)

train.to_csv("data/train.csv", index=False)
dev.to_csv(  "data/dev.csv",   index=False)
test.to_csv( "data/test.csv",  index=False)

print("Splits:", len(train), len(dev), len(test))
