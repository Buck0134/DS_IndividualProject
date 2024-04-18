from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from joblib import dump, load

class TextClassifier:
    def __init__(self, model_type='random_forest'):
        self.model = self._initialize_model(model_type)

    def _initialize_model(self, model_type):
        if model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic_regression':
            return LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'svm':
            return SVC(kernel='linear', random_state=42)
        else:
            raise ValueError("Unsupported model type provided. Choose 'random_forest', 'logistic_regression', or 'svm'.")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("Training complete.")

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
    
    def get_model(self):
        return self.model

    def check_for_overfitting(self):
        """
        Evaluates the model for signs of overfitting by comparing F1 scores on the training and test sets.
        """
        # Predict on training and test data
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        # Calculate F1 score for training and test sets
        f1_train = f1_score(self.y_train, y_train_pred, average='macro')
        f1_test = f1_score(self.y_test, y_test_pred, average='macro')

        print(f"Training F1 Score: {f1_train}")
        print(f"Test F1 Score: {f1_test}")

        # Check the difference
        if f1_train > f1_test:
            print("Model may be overfitting.")
        else:
            print("Model performance is consistent across training and test sets.")
    def load_model(self, path):
        self.model = load(path)
        print("Model loaded successfully")
    
    def save_model(self, path):
        # Use joblib to save the model to a file
        dump(self.model, path)
        print(f"Model saved to {path}")
