import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils import shuffle
import pickle
import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re

# Assuming preprocessing, vectorizing, and loading data is already handled as per your existing setup

nltk.download('punkt')
nltk.download('stopwords')
def clean_text(text):
    # Decode bytes to string if necessary
    if text.startswith("b'") or text.startswith('b"'):
        text = eval(text)  # converts bytes string to actual bytes
        text = text.decode('utf-8')  # decodes bytes to string

    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()

    # Normalize text
    text = text.lower()

    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Join words back to form the cleaned text
    return ' '.join(filtered_tokens)

# Load vectorizer and transformer
with open('count_vectorizer.pkl', 'rb') as file:
    bow_vectorizer = pickle.load(file)

with open('tfidf_transformer.pkl', 'rb') as file:
    tfidf_transformer = pickle.load(file)

# Load and shuffle data
df = pd.read_csv('data/comments.csv')
print(df.describe())
# bot_df = df[df['label'] == 'Human']  # DataFrame with only 'Bot' labeled rows
bot_df = shuffle(df, random_state=42)
bot_df['comment'] = bot_df['comment'].apply(clean_text)

bow_features = bow_vectorizer.transform(bot_df['comment'])
tfidf_features = tfidf_transformer.transform(bow_features)

# Define range of k to try
k_values = range(1, 10)  # This range can be adjusted based on specific needs
inertia = []

# Calculate inertia for each k
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(tfidf_features)
    inertia.append(kmeans.inertia_)

# Calculate the rate of change of inertia
inertia_change = np.diff(inertia) / inertia[:-1] * 100  # percentage change from one k to the next

# Plot inertia and inertia rate of change
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Number of clusters (k)')
ax1.set_ylabel('Inertia', color=color)
ax1.plot(k_values, inertia, 'o-', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Rate of change of inertia (%)', color=color)
ax2.plot(k_values[1:], -inertia_change, 'o-', color=color)  # Negative change to show decrease
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Elbow Method and Rate of Change of Inertia')
plt.grid(True)
plt.show()