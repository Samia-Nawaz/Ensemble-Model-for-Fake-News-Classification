
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords, wordnet

nltk.download('stopwords')
nltk.download('sentiwordnet')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# Load the dataset
def load_dataset(path):
    df = pd.read_csv(path)
    X = df['text']
    y = df['label']
    return X, y


def apply_lexicon_mpqa(text_data):
    mpqa_lexicon = {'good': 1, 'bad': -1, 'neutral': 0}
    features = []
    for text in text_data:
        score = sum([mpqa_lexicon.get(word, 0) for word in text.split()])
        features.append([score])
    return np.array(features)


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def apply_lexicon_swn(text_data):
    features = []
    for text in text_data:
        score = 0
        words = text.split()
        for word in words:
            word_pos = get_wordnet_pos(word)
            synsets = list(swn.senti_synsets(word, pos=word_pos))
            if synsets:
                synset = synsets[0]
                score += synset.pos_score() - synset.neg_score()
        features.append([score])
    return np.array(features)


def apply_lexicon_vsl(text_data):
    vsl_lexicon = {'not': -1, 'very': 1.5, 'barely': -0.5}
    features = []
    for text in text_data:
        score = 0
        words = text.split()
        for i, word in enumerate(words):
            if word in vsl_lexicon:
                score += vsl_lexicon[word]
        features.append([score])
    return np.array(features)


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
    # Load the dataset
    X, y = load_dataset('fake_news_data.csv')

    # Apply lexicon-based features
    X_mpqa = apply_lexicon_mpqa(X)
    X_swn = apply_lexicon_swn(X)
    X_vsl = apply_lexicon_vsl(X)

    # Combine lexicon features
    X_combined = np.hstack((X_mpqa, X_swn, X_vsl))

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    # Train classifiers
    train_classifiers(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
