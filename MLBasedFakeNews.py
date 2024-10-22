import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')


def load_dataset(path):
    df = pd.read_csv(path)
    X = df['text']
    y = df['label']
    return X, y


# Preprocess text data using CountVectorizer
def preprocess_text(text_data):
    stop_words = set(stopwords.words('english'))
    vectorizer = CountVectorizer(stop_words=stop_words)
    return vectorizer.fit_transform(text_data)


# Train and evaluate classifiers
def train_classifiers(X_train, X_test, y_train, y_test):
    classifiers = {
        'DecisionTree': DecisionTreeClassifier(),
        'SVM': SVC(probability=True),
        'RandomForest': RandomForestClassifier()
    }

    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} Accuracy: {acc}")
    return results


# Main function to execute the process
def main():
    X, y = load_dataset('fake_news_data.csv')
    X = preprocess_text(X)

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train classifiers
    train_classifiers(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
