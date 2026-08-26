import pandas as pd
df = pd.read_parquet("results/quantitative-evals/passk_qwen3.5-27b.parquet")
print(df.head(20))
print(df.shape)
# e.g. reproduce the headline yourself:
print((df["rank"] <= 10).groupby([df["set"], df["lens"]]).mean().unstack())