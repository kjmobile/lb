#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# 원본 데이터 읽기
df = pd.read_csv('flights.csv')

print(f"Original dataset shape: {df.shape}")
print(f"Original cancellation rate: {df['CANCELLED'].mean():.4f}")
print(f"Cancelled flights: {df['CANCELLED'].sum()}")
print(f"Not cancelled flights: {(df['CANCELLED']==0).sum()}")

# Cancelled와 Not Cancelled 분리
df_cancelled = df[df['CANCELLED'] == 1]
df_not_cancelled = df[df['CANCELLED'] == 0]

# 샘플링 (총 10,000개)
# Option 1: 20% cancellation rate
n_cancelled = 1450
n_not_cancelled = 8550

# Cancelled flights에서 2000개 샘플링 (있으면)
if len(df_cancelled) >= n_cancelled:
    sample_cancelled = df_cancelled.sample(n=n_cancelled, random_state=42)
else:
    print(f"Warning: Only {len(df_cancelled)} cancelled flights available")
    sample_cancelled = df_cancelled.sample(n=len(df_cancelled), random_state=42)
    n_cancelled = len(sample_cancelled)
    n_not_cancelled = 10000 - n_cancelled

# Not cancelled flights에서 8000개 샘플링
sample_not_cancelled = df_not_cancelled.sample(n=n_not_cancelled, random_state=42)

# 합치기
df_sample = pd.concat([sample_cancelled, sample_not_cancelled], ignore_index=True)

# 섞기
df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)

# 저장
df_sample.to_csv('flights_s.csv', index=False)

print(f"\n=== New Dataset ===")
print(f"Shape: {df_sample.shape}")
print(f"Cancellation rate: {df_sample['CANCELLED'].mean():.4f}")
print(f"Cancelled flights: {df_sample['CANCELLED'].sum()}")
print(f"Not cancelled flights: {(df_sample['CANCELLED']==0).sum()}")
print(f"\nSaved to flights_s.csv")


# In[ ]:




