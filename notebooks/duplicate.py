import pandas as pd

df = pd.read_parquet("data/bronze/gtex/gene_tpm_raw.parquet", columns=None)
print(df.columns[:5].tolist())