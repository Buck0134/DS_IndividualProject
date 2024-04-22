import pandas as pd
from preprocessing.preprocess import clean_text
from textProcessor.textProcessor import TextPreprocessor
import numpy as np
from sklearn.utils import shuffle
from Models.textClassifier import TextClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score


def perform_cross_validation(model, features, labels, num_folds=5):
    """
    Performs K-Fold cross-validation on the provided dataset using the specified model.

    Args:
        model: The machine learning model to use.
        features (array-like): The input features for the model.
        labels (array-like): The labels corresponding to the input features.
        num_folds (int): The number of folds to use in the cross-validation.

    Returns:
        dict: A dictionary containing the cross-validation scores and statistics.
    """
    # Ensure balanced class distribution within each fold
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score, average='binary', pos_label = 'Human')  # Corrected scorer
    scores = cross_val_score(model, features, labels, cv=kf, scoring=f1_scorer)
    return {
        "scores": scores,
        "average_score": scores.mean(),
        "std_deviation": scores.std()
    }

# Making sure the dataset is okay to use.
df = pd.read_csv('data/comments.csv')

# print(df.head)
# print(df.shape)
has_nan = df.isna().any().any()
if has_nan:
    print("The dataset has NaN values, needs further processing")
else:
    print("The dataset has no NaN values! Seems to be perfect")


df['comment'] = df['comment'].apply(clean_text)
df = shuffle(df, random_state=42)
# print(df['comment'].head(10))

vect_method = ['tfidf', 'bow', 'embeddings']

for each_vect in vect_method: 
    print(f'\n\nVectorized by {each_vect}')
    preprocessor = TextPreprocessor()

    features = preprocessor.apply_vectorization(df, 'comment', method=each_vect)

    # Now, save both the CountVectorizer and the TfidfTransformer
    with open('count_vectorizer.pkl', 'wb') as file:
        pickle.dump(preprocessor.bow_vectorizer, file)

    with open('tfidf_transformer.pkl', 'wb') as file:
        pickle.dump(preprocessor.tfidf_transformer, file)

    # if isinstance(features, np.ndarray):
    #     # Assuming features are from embeddings and you want to add them as separate columns
    #     for i in range(features.shape[1]):
    #         df[f'feature_{i}'] = features[:, i]

    labels = df['label']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Training data ready
    model_list = ['random_forest', 'logistic_regression', 'svm']
    for model in model_list:
        print(f"Training for {model}")
        Classifier = TextClassifier(model_type=model)
        Classifier.train(X_train, y_train)
        # print(f"Model Result")
        # Classifier.evaluate(X_test, y_test)
        # Classifier.check_for_overfitting(X_train, y_train, X_test, y_test)
        results = perform_cross_validation(Classifier.get_model(), features, labels, num_folds=5)
        print("F1 Scores for each fold:", results["scores"])
        print("Average F1 Score:", results["average_score"])
        print("Standard Deviation of F1 Scores:", results["std_deviation"])




