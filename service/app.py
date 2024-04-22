import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from flask import Flask, request, jsonify
import requests
from joblib import load
import pandas as pd
import pickle
from preprocessing.preprocess import clean_text
from textProcessor.textProcessor import TextPreprocessor
import csv
import hashlib
import os

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

def hash_username(username):
    # Create a hash object
    hasher = hashlib.sha256()
    hasher.update(username.encode('utf-8'))
    # Return the hex digest of the username
    return hasher.hexdigest()

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the application!"

# Load the Random Forest model
production_model = load('service/random_forest_model.joblib')

# Load the CountVectorizer
with open('count_vectorizer.pkl', 'rb') as file:
    bow_vectorizer = pickle.load(file)

# Load the TfidfTransformer
with open('tfidf_transformer.pkl', 'rb') as file:
    tfidf_transformer = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    username = data['username'] 
    comment_text = data['comment']

    # Create DataFrame from the incoming text
    df = pd.DataFrame({'comment': [comment_text]})
    df['comment'] = df['comment'].apply(clean_text)

    preprocessor = TextPreprocessor() 
    features = preprocessor.apply_vectorization(df, 'comment', method='embeddings')
    # # Transform the text to a bag-of-words matrix and apply TF-IDF
    # bow_features = bow_vectorizer.transform(df['comment'])
    # tfidf_features = tfidf_transformer.transform(bow_features)

    # Predict using the loaded model
    predictions = production_model.predict(features)

    # Return predictions along with username and prediction result
    return jsonify({'username': username, 'predictions': predictions.tolist()})

# Changed from GET to POST because we are sending potentially sensitive data in the request body, which is more secure than URL parameters.
@app.route('/api/pr_comments', methods=['POST'])
def pr_comments():
    data = request.get_json()
    if not data or 'repo' not in data or 'pr_number' not in data or 'token' not in data:
        return jsonify({'error': 'Missing required parameters'}), 400

    repo = data['repo']
    pr_number = data['pr_number']
    token = data['token']

    comments = get_comments(repo, pr_number, token)
    filtered_comments = [{
        'username': comment['user']['login'],
        'comment': comment['body'],
        'label': 'Bot'
    } for comment in comments if comment['user']['login'] == 'github-actions[bot]']

    return jsonify(filtered_comments)

def get_comments(repo, pr_number, token):
    GITHUB_API_URL = 'https://api.github.com'
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    url = f'{GITHUB_API_URL}/repos/{repo}/issues/{pr_number}/comments'
    response = requests.get(url, headers=headers)
    response.raise_for_status()  # Raises an HTTPError for bad responses
    return response.json()

if __name__ == '__main__':
    # Downlading punkt and stopwords resources. 
    nltk.download('punkt')
    nltk.download('stopwords')
    app.run(debug=True, host='0.0.0.0', port=5001)

