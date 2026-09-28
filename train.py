import os
import re
import glob
import joblib
import pandas as pd
import kagglehub

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# ---------------------------------------------------------
# Step 1: Download & Load Kaggle Sentiment140 Dataset
# ---------------------------------------------------------
print("Downloading Sentiment140 dataset via kagglehub...")
path = kagglehub.dataset_download("kazanova/sentiment140")
print("Path to dataset files:", path)

# Locate the CSV file inside the downloaded directory
csv_files = glob.glob(os.path.join(path, "*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV files found in the downloaded dataset folder.")

dataset_path = csv_files[0]
print(f"Loading data from: {dataset_path}")

# Sentiment140 schema: Column 0 = Target (0: Neg, 4: Pos), Column 5 = Text
columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
raw_df = pd.read_csv(dataset_path, encoding='ISO-8859-1', names=columns)

# Keep relevant columns and remap sentiment target (4 -> 1)
df = raw_df[['text', 'target']].copy()
df['sentiment'] = df['target'].map({0: 0, 4: 1})

# Sample dataset for fast training (adjust n_samples as needed)
SAMPLE_SIZE = 50000
df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
print(f"Dataset loaded successfully with {len(df)} sampled records.")

# ---------------------------------------------------------
# Step 2: Text Preprocessing
# ---------------------------------------------------------
stop_words = set(stopwords.words('english'))

def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'(@\w+|#\w+)', '', text)                               # Remove mentions/hashtags
    text = re.sub(r'[^\w\s]', '', text)                                  # Remove punctuation
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

print("Cleaning tweets...")
df['clean_text'] = df['text'].apply(clean_tweet)

# ---------------------------------------------------------
# Step 3: Feature Extraction (TF-IDF)
# ---------------------------------------------------------
print("Vectorizing text data...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# Step 4: Model Training & Evaluation
# ---------------------------------------------------------
print("Training Logistic Regression Model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# ---------------------------------------------------------
# Step 5: Save Model Artifacts
# ---------------------------------------------------------
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vector.pkl')
print("Model and vectorizer saved successfully as `sentiment_model.pkl` and `tfidf_vector.pkl`.")