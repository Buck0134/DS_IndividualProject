import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

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

from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
import pickle
from preprocessing.preprocess import clean_text
import csv
import hashlib
import os

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
    user_type = data['user_type']
    
    if user_type.lower() == "bot":
        correct_result = "Bot"
    else:
        correct_result = "Human"

    # Create DataFrame from the incoming text
    df = pd.DataFrame({'comment': [comment_text]})
    df['comment'] = df['comment'].apply(clean_text)

    # Transform the text to a bag-of-words matrix and apply TF-IDF
    bow_features = bow_vectorizer.transform(df['comment'])
    tfidf_features = tfidf_transformer.transform(bow_features)

    # Predict using the loaded model
    predictions = production_model.predict(tfidf_features)

    file_path = 'new_data.csv'
    header = ['username', 'comment', 'label']
    data = [hash_username(username), comment_text, correct_result]
    if not os.path.isfile(file_path) or os.stat(file_path).st_size == 0:
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)  # Write the header if file doesn't exist or is empty
            writer.writerow(data)
    else:
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)

    # Return predictions along with username and prediction result
    return jsonify({'username': username, 'predictions': predictions.tolist(), 'type': correct_result})


if __name__ == '__main__':
    # Downlading punkt and stopwords resources. 
    nltk.download('punkt')
    nltk.download('stopwords')
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)

