# from sklearn.cluster import KMeans
# from bs4 import BeautifulSoup
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# import nltk
# import re
# import pickle
# import pandas as pd
# from sklearn.utils import shuffle
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# # Assuming your vectorized data is stored in a variable named `X`
# nltk.download('punkt')
# nltk.download('stopwords')
# def clean_text(text):
#     # Decode bytes to string if necessary
#     if text.startswith("b'") or text.startswith('b"'):
#         text = eval(text)  # converts bytes string to actual bytes
#         text = text.decode('utf-8')  # decodes bytes to string

#     # Remove HTML tags
#     text = BeautifulSoup(text, 'html.parser').get_text()

#     # Normalize text
#     text = text.lower()

#     # Remove special characters and punctuation
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

#     # Tokenize text
#     tokens = word_tokenize(text)

#     # Remove stop words
#     stop_words = set(stopwords.words('english'))
#     filtered_tokens = [token for token in tokens if token not in stop_words]

#     # Join words back to form the cleaned text
#     return ' '.join(filtered_tokens)

# # Load the CountVectorizer
# with open('count_vectorizer.pkl', 'rb') as file:
#     bow_vectorizer = pickle.load(file)

# # Load the TfidfTransformer
# with open('tfidf_transformer.pkl', 'rb') as file:
#     tfidf_transformer = pickle.load(file)

# df = pd.read_csv('data/comments.csv')
# df = shuffle(df, random_state=42)

# bow_features = bow_vectorizer.transform(df['comment'])
# tfidf_features = tfidf_transformer.transform(bow_features)
# # Choose the number of clusters
# k = 5  # Adjust this based on your specific dataset or use the Elbow method to find the optimal k

# # Initialize and fit the K-means model
# kmeans = KMeans(n_clusters=k, random_state=42)
# kmeans.fit(tfidf_features)

# # Get the cluster labels for each data point
# cluster_labels = kmeans.labels_

# # Optionally, calculate centroids
# centroids = kmeans.cluster_centers_

# # # Print the cluster labels to see the distribution
# # print("Cluster labels:")
# # print(cluster_labels)

# # # If you want to examine centroids (the mean vector of each cluster)
# # print("Centroids:")
# # print(centroids)

# # Reduce dimensions for visualization
# pca = PCA(n_components=2)
# reduced_features = pca.fit_transform(tfidf_features.toarray())  # Convert sparse matrix to dense if needed for PCA

# # Plot the clusters
# plt.figure(figsize=(10, 6))
# colors = ['r', 'g', 'b', 'y', 'c']  # Colors for clusters 0-4
# for i in range(k):
#     plt.scatter(reduced_features[cluster_labels == i, 0], reduced_features[cluster_labels == i, 1], color=colors[i], label=f'Cluster {i}', alpha=0.7)
# plt.title('2D Visualization of Comments Clusters')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.legend()
# plt.grid(True)
# plt.show()


from sklearn.cluster import KMeans
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
import pickle
import pandas as pd
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
import numpy as np

# Assuming your vectorized data is stored in a variable named `X`
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

# Load the CountVectorizer
with open('count_vectorizer.pkl', 'rb') as file:
    bow_vectorizer = pickle.load(file)

# Load the TfidfTransformer
with open('tfidf_transformer.pkl', 'rb') as file:
    tfidf_transformer = pickle.load(file)

df = pd.read_csv('data/comments.csv')
bot_df = df[df['label'] == 'Bot']  # DataFrame with only 'Bot' labeled rows
bot_df = shuffle(bot_df, random_state=42)

bow_features = bow_vectorizer.transform(bot_df['comment'])
tfidf_features = tfidf_transformer.transform(bow_features)
# Choose the number of clusters
# k = 5  # Adjust this based on your specific dataset or use the Elbow method to find the optimal k

# Analyze clusters from k = 1 to 7
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(tfidf_features)
    
    cluster_labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    
    if k > 1:  # Silhouette score is only valid when k > 1
        silhouette_avg = silhouette_score(tfidf_features, cluster_labels)
    else:
        silhouette_avg = None
    
    # Calculate separation
    if k > 1:
        dist_matrix = squareform(pdist(centroids, 'euclidean'))
        separation = np.sum(np.min(dist_matrix[np.nonzero(dist_matrix)], axis=0))
    else:
        separation = None

    # Print cluster information
    cluster, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = {c: count for c, count in zip(cluster, counts)}

    print(f"\nFor k={k}:")
    print(f"  Cohesion (Total Inertia): {inertia:.3f}")
    print(f"  Silhouette Score: {silhouette_avg:.3f}" if silhouette_avg is not None else "  Silhouette Score: N/A")
    print(f"  Separation: {separation:.3f}" if separation is not None else "  Separation: N/A")
    print(f"  Cluster Sizes: {cluster_sizes}")
# # Initialize and fit the K-means model
# kmeans = KMeans(n_clusters=k, random_state=42)
# kmeans.fit(tfidf_features)

# # Get the cluster labels for each data point
# cluster_labels = kmeans.labels_

# # Optionally, calculate centroids
# centroids = kmeans.cluster_centers_

# # # Print the cluster labels to see the distribution
# # print("Cluster labels:")
# # print(cluster_labels)

# # # If you want to examine centroids (the mean vector of each cluster)
# # print("Centroids:")
# # print(centroids)

# # Reduce dimensions for visualization
# pca = PCA(n_components=2)
# reduced_features = pca.fit_transform(tfidf_features.toarray())  # Convert sparse matrix to dense if needed for PCA

# # Plot the clusters
# plt.figure(figsize=(10, 6))
# colors = ['r', 'g', 'b', 'y', 'c']  # Colors for clusters 0-4
# for i in range(k):
#     plt.scatter(reduced_features[cluster_labels == i, 0], reduced_features[cluster_labels == i, 1], color=colors[i], label=f'Cluster {i}', alpha=0.7)
# plt.title('2D Visualization of Comments Clusters')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.legend()
# plt.grid(True)
# plt.show()

