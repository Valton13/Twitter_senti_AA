import os
import glob
import pandas as pd
import kagglehub

# Directory to save formatted batch files
OUTPUT_DIR = "test_batch_datasets"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List of Kaggle datasets with target text column configuration
datasets_config = [
    {
        "name": "Twitter Airline Sentiment",
        "slug": "crowdflower/twitter-airline-sentiment",
        "text_col": "text",
        "output_file": "01_airline_tweets.csv"
    },
    {
        "name": "Financial News Headlines",
        "slug": "ankurzing/sentiment-analysis-for-financial-news",
        "text_col": "Sentence",
        "output_file": "02_financial_news.csv"
    },
    {
        "name": "IMDB Movie Reviews",
        "slug": "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
        "text_col": "review",
        "output_file": "03_imdb_reviews.csv"
    },
    {
        "name": "ChatGPT Twitter Sentiment",
        "slug": "shenba/chatgpt-twitter-dataset",
        "text_col": "tweets",
        "output_file": "04_chatgpt_tweets.csv"
    }
]

print("Downloading and preparing sample batch test files...\n")

for config in datasets_config:
    try:
        print(f"Fetching: {config['name']}...")
        path = kagglehub.dataset_download(config["slug"])
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        
        if csv_files:
            # Read first CSV
            df = pd.read_csv(csv_files[0], encoding='latin-1', low_memory=False)
            
            # Locate text column
            col_match = [c for c in df.columns if config["text_col"].lower() in c.lower()]
            
            if col_match:
                target_col = col_match[0]
                sample_df = pd.DataFrame({'tweet': df[target_col].dropna().sample(n=min(500, len(df)), random_state=42)})
                
                # Save formatted CSV
                save_path = os.path.join(OUTPUT_DIR, config["output_file"])
                sample_df.to_csv(save_path, index=False)
                print(f"Saved: `{save_path}` with {len(sample_df)} records.\n")
            else:
                print(f"Could not find matching text column for {config['name']}.\n")
    except Exception as e:
        print(f"Skipped {config['name']} due to error: {e}\n")

print(f"All sample datasets are saved in the `{OUTPUT_DIR}/` directory. You can drop any of these files into the Streamlit Batch Custom CSV Analytics tab!")