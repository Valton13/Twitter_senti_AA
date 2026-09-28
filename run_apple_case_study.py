import os
import re
import glob
import joblib
import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# NLTK Dependencies
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
stop_words = set(stopwords.words('english'))

# ---------------------------------------------------------
# 1. Download & Prepare Apple Inc. Tweet Dataset
# ---------------------------------------------------------
print("Downloading Apple Public Tweets Dataset from Kaggle...")
path = kagglehub.dataset_download("slythe/apple-twitter-sentiment-crowdflower")

# Locate CSV file
csv_files = glob.glob(os.path.join(path, "*.csv"))
if not csv_files:
    raise FileNotFoundError("Could not locate Apple dataset CSV.")

raw_df = pd.read_csv(csv_files[0], encoding='ISO-8859-1')

# Extract text column ('text' or 'text_clean')
text_column = [col for col in raw_df.columns if 'text' in col.lower()][0]
apple_df = pd.DataFrame({'tweet': raw_df[text_column].dropna()})

print(f"Loaded {len(apple_df)} real-world public tweets about Apple Inc.")

# ---------------------------------------------------------
# 2. Preprocessing Pipeline
# ---------------------------------------------------------
def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r'(@\w+|#\w+)', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(filtered)

print("Preprocessing raw tweets...")
apple_df['clean_tweet'] = apple_df['tweet'].apply(clean_tweet)

# Filter out empty cleaned strings
apple_df = apple_df[apple_df['clean_tweet'].str.strip() != ''].reset_index(drop=True)

# ---------------------------------------------------------
# 3. Model Inference & Prediction
# ---------------------------------------------------------
print("Loading Sentiment Model & Vectorizer...")
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vector.pkl')

print("Generating predictions...")
X_apple = vectorizer.transform(apple_df['clean_tweet'])
apple_df['pred_code'] = model.predict(X_apple)
probabilities = model.predict_proba(X_apple)

apple_df['sentiment'] = apple_df['pred_code'].map({1: 'Positive', 0: 'Negative'})
apple_df['confidence'] = [
    probs[1] if code == 1 else probs[0] 
    for code, probs in zip(apple_df['pred_code'], probabilities)
]

# ---------------------------------------------------------
# 4. Generate Visual Analytics
# ---------------------------------------------------------
print("Generating visualization charts...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Chart 1: Sentiment Distribution
pos_count = (apple_df['pred_code'] == 1).sum()
neg_count = (apple_df['pred_code'] == 0).sum()

axes[0].pie(
    [pos_count, neg_count], 
    labels=['Positive', 'Negative'], 
    autopct='%1.1f%%', 
    colors=['#2ecc71', '#e74c3c'], 
    startangle=90,
    explode=(0.05, 0)
)
axes[0].set_title('Apple Inc. Public Sentiment Distribution', fontsize=14, fontweight='bold')

# Chart 2: Top Word Drivers (Positive vs Negative)
pos_text = " ".join(apple_df[apple_df['pred_code'] == 1]['clean_tweet'])
neg_text = " ".join(apple_df[apple_df['pred_code'] == 0]['clean_tweet'])

if pos_text.strip():
    wc = WordCloud(width=400, height=300, background_color='white', colormap='Greens').generate(pos_text)
    axes[1].imshow(wc, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title('Top Drivers of Positive Sentiment', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('apple_sentiment_summary.png', dpi=300)
print("Saved visualization summary to `apple_sentiment_summary.png`.")

# Save dataset with predictions
apple_df[['tweet', 'sentiment', 'confidence']].to_csv('apple_sentiment_results.csv', index=False)
print("Saved predictions to `apple_sentiment_results.csv`.")