import pandas as pd
import os
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def prepare_kvasir_metadata():
    os.makedirs('data', exist_ok=True)
    print("Downloading Kvasir-Capsule official splits...")
    
    # Kvasir GitHub provides two splits. We combine them to form our full metadata.
    df0 = pd.read_csv("https://raw.githubusercontent.com/simula/kvasir-capsule/master/official_splits/split_0.csv")
    df1 = pd.read_csv("https://raw.githubusercontent.com/simula/kvasir-capsule/master/official_splits/split_1.csv")
    
    df = pd.concat([df0, df1], ignore_index=True)
    
    # Rename 'label' to 'class' to match our existing EndoClean pipeline
    df = df.rename(columns={'label': 'class'})
    
    # Kvasir filenames are formatted as: videoID_frameIndex.jpg (e.g., bca26705313a4644_20404.jpg)
    # We extract the video_id for patient-wise cross-validation
    df['video_id'] = df['filename'].apply(lambda x: x.split('_')[0])
    
    df.to_csv("data/metadata.csv", index=False)
    print(f"Successfully generated data/metadata.csv with {len(df)} records!")

if __name__ == "__main__":
    prepare_kvasir_metadata()
