import pandas as pd
from pathlib import Path

# Combine split_0 and split_1 to make our full metadata.csv
df0 = pd.read_csv("https://raw.githubusercontent.com/simula/kvasir-capsule/master/official_splits/split_0.csv")
df1 = pd.read_csv("https://raw.githubusercontent.com/simula/kvasir-capsule/master/official_splits/split_1.csv")

df = pd.concat([df0, df1])

# Rename 'label' to 'class' to match our code
df = df.rename(columns={'label': 'class'})

# Extract video_id from filename (format: videoId_frameNumber.jpg)
df['video_id'] = df['filename'].apply(lambda x: x.split('_')[0])

df.to_csv("data/metadata.csv", index=False)
print("Saved clean metadata to data/metadata.csv")
