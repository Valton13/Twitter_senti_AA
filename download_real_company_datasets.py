import os
import glob
import pandas as pd
import kagglehub

OUTPUT_DIR = "company_batch_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset configurations
company_datasets = [
    {
        "company": "US_Airlines",
        "slug": "crowdflower/twitter-airline-sentiment",
        "text_col": "text",
        "filename": "us_airlines_tweets.csv"
    },
    {
        "company": "ChatGPT_OpenAI",
        "slug": "shenba/chatgpt-twitter-dataset",
        "text_col": "tweets",
        "filename": "chatgpt_public_tweets.csv"
    },
    {
        "company": "McDonalds",
        "slug": "raghadalharbi/mcdonalds-store-reviews",
        "text_col": "review",
        "filename": "mcdonalds_store_reviews.csv"
    }
]

print("Downloading and preparing company batch test datasets...\n")

for item in company_datasets:
    try:
        print(f"Fetching {item['company']} dataset...")
        path = kagglehub.dataset_download(item["slug"])
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        
        if csv_files:
            df = pd.read_csv(csv_files[0], encoding='latin-1', low_memory=False)
            
            # Find matching text column
            col_match = [c for c in df.columns if item["text_col"].lower() in c.lower()]
            
            if col_match:
                target_col = col_match[0]
                # Sample 500 records for fast batch testing
                sample_df = pd.DataFrame({'tweet': df[target_col].dropna().sample(n=min(500, len(df)), random_state=42)})
                
                save_path = os.path.join(OUTPUT_DIR, item["filename"])
                sample_df.to_csv(save_path, index=False)
                print(f"Successfully saved `{save_path}` ({len(sample_df)} records).\n")
            else:
                print(f"Text column '{item['text_col']}' not found.\n")
    except Exception as e:
        print(f"Error fetching {item['company']}: {e}\n")

print(f"Done! All datasets are ready in the `{OUTPUT_DIR}/` folder. Drop any file into the Streamlit Batch Analytics tab.")