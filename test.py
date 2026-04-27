import pandas as pd

df = pd.read_parquet('data/processed/train_data.parquet')
print(df.sample(10))
print(df.sample(10).isnull().sum())
print(df.shape)
print(df['AttackCategory'].value_counts())