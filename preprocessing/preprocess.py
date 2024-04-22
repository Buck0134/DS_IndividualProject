# import pandas as pd

# # Making sure the dataset is okay to use.

# df = pd.read_csv('data/comments.csv')

# print(df.head)
# # print(df.shape)
# has_nan = df.isna().any().any()
# if has_nan:
#     print("The dataset has NaN values, needs further processing")
# else:
#     print("The dataset has no NaN values! Seems to be perfect")


# print(df['label'].describe())

# Start Data preprocessing for natual language processing
# We are using beautiful soup for removing html tags and nltk for tokenization
import re
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Downlading punkt and stopwords resources. 
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

# # Apply cleaning function to the comments
# df['comment'] = df['comment'].apply(clean_text)

# print(df.head())