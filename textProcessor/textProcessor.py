import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import spacy

class TextPreprocessor:
    def __init__(self):
        # Load pre-trained language model from spaCy
        self.bow_vectorizer = CountVectorizer()
        self.tfidf_transformer = TfidfTransformer()
        self.nlp = spacy.load('en_core_web_md')

    def vectorize_bow(self, data):
        bow_matrix = self.bow_vectorizer.fit_transform(data)
        return bow_matrix

    def vectorize_tfidf(self, data):
        # First, transform the data to a bag-of-words matrix
        bow_matrix = self.vectorize_bow(data)
        # Then, apply TF-IDF transformation
        tfidf_matrix = self.tfidf_transformer.fit_transform(bow_matrix)
        return tfidf_matrix

    def vectorize_embeddings(self, data):
        """
        Vectorize data using spaCy to generate document vectors by averaging word vectors.

        Args:
            data (pd.Series): A series of text documents.

        Returns:
            np.ndarray: A numpy array where each row represents the vectorized form of a document.
        """
        def document_vector(doc):
            doc = self.nlp(doc)
            vectors = [token.vector for token in doc if not token.is_stop and not token.is_punct and token.has_vector]
            if vectors:
                return np.mean(vectors, axis=0)
            else:
                return np.zeros(self.nlp.vocab.vectors_length)  # Return a zero vector if doc has no tokens or all tokens are stopwords/punctuations
        
        return np.vstack(data.apply(document_vector))

    def apply_vectorization(self, df, text_column, method='tfidf'):
        """
        Apply vectorization method to a specific column in a DataFrame and return the feature matrix.

        Args:
        df (pd.DataFrame): DataFrame containing the text data.
        text_column (str): Column name containing the text data.
        method (str): Method of vectorization ('bow', 'tfidf', 'embeddings').

        Returns:
        Depending on the method, returns a sparse matrix (BoW, TF-IDF) or a numpy array (embeddings).
        """
        text_data = df[text_column]
        if method == 'bow':
            return self.vectorize_bow(text_data)
        elif method == 'tfidf':
            return self.vectorize_tfidf(text_data)
        elif method == 'embeddings':
            return self.vectorize_embeddings(text_data)
        else:
            raise ValueError("Unsupported method specified. Choose 'bow', 'tfidf', or 'embeddings'.")
