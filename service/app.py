from joblib import load
import pandas as pd
from textProcessor.textProcessor import TextPreprocessor
from preprocessing.preprocess import clean_text
import pickle

production_model = load('service/random_forest_model.joblib')

with open('text.txt', 'r') as file:
    # Read the content of the file into a string
    file_content = file.read()

# Create a DataFrame with one row and one column
df = pd.DataFrame({'comment': [file_content]}, index=[1])
df['comment'] = df['comment'].apply(clean_text)

with open('count_vectorizer.pkl', 'rb') as file:
    bow_vectorizer = pickle.load(file)

# Load the TfidfTransformer
with open('tfidf_transformer.pkl', 'rb') as file:
    tfidf_transformer = pickle.load(file)

# Assuming you have new text data in 'new_df' under the 'comment' column
# First, transform the text data to a bag-of-words matrix
bow_features = bow_vectorizer.transform(df['comment'])

# Then, apply TF-IDF transformation
tfidf_features = tfidf_transformer.transform(bow_features)

# preprocessor = TextPreprocessor()
# new_features = preprocessor.apply_vectorization(df, 'comment', method='embeddings')
predictions = production_model.predict(tfidf_features)
print("Predictions:", predictions)