import pandas as pd
df = pd.read_parquet("results/quantitative-evals/passk_gemma-3-27b-it.parquet")
print(df.head(20))
print(df.shape)
# e.g. reproduce the headline yourself:
print((df["rank"] <= 10).groupby([df["set"], df["lens"]]).mean().unstack())