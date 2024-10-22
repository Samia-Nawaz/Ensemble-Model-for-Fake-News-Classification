import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
import nltk
from nltk.corpus import stopwords


nltk.download('mpqa')
nltk.download('sentiwordnet')
nltk.download('vader_lexicon')

def load_dataset(path):
    df = pd.read_csv(path)
    X = df['text']
    y = df['label']
    return X, y


X, y = load_dataset('fake_news_data.csv')



# Preprocess the text data
def preprocess_text(text_data):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    vectorizer = CountVectorizer(stop_words=stop_words)
    return vectorizer.fit_transform(text_data)




def apply_lexicon_mpqa(text_data):
    mpqa_lexicon = {
        'good': 1, 'bad': -1, 'happy': 1, 'sad': -1, 'neutral': 0
    }
    features = []
    for text in text_data:
        score = 0
        words = text.split()
        for word in words:
            if word in mpqa_lexicon:
                score += mpqa_lexicon[word]
        features.append([score])
    return np.array(features)


from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet


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


# Placeholder function for VSL-SWC lexicon scoring
def apply_lexicon_vsl(text_data):
    vsl_lexicon = {
        'not': -1, 'very': 1.5, 'barely': -0.5  # Example entries
        # Extend this dictionary with full VSL lexicon
    }

    features = []
    for text in text_data:
        score = 0
        words = text.split()
        for i, word in enumerate(words):
            if word in vsl_lexicon:
                if i + 1 < len(words):  # Look at the next word
                    next_word = words[i + 1]
                    if next_word in vsl_lexicon:
                        score += vsl_lexicon[word] * vsl_lexicon[next_word]
                    else:
                        score += vsl_lexicon[word]
        features.append([score])  # You can add more complex feature representations
    return np.array(features)


# Train and evaluate classifiers for each lexicon
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
        results[name] = (clf, acc, y_pred)
        print(f"{name} Accuracy: {acc}")

    return results


# Custom Classifier IBA-ENet
def build_iba_enet(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification output
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Main function to execute the process
def main():
    # Load and preprocess data
    X, y = load_dataset('fake_news_data.csv')
    X = preprocess_text(X)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply Lexicons
    X_train_mpqa = apply_lexicon_mpqa(X_train)
    X_test_mpqa = apply_lexicon_mpqa(X_test)

    X_train_swn = apply_lexicon_swn(X_train)
    X_test_swn = apply_lexicon_swn(X_test)

    X_train_vsl = apply_lexicon_vsl(X_train)
    X_test_vsl = apply_lexicon_vsl(X_test)

    # Train classifiers for MPQA-SWC
    print("Training classifiers for MPQA-SWC:")
    mpqa_results = train_classifiers(X_train_mpqa, X_test_mpqa, y_train, y_test)

    # Train classifiers for SWN-SWC
    print("Training classifiers for SWN-SWC:")
    swn_results = train_classifiers(X_train_swn, X_test_swn, y_train, y_test)

    # Train classifiers for VSL-SWC
    print("Training classifiers for VSL-SWC:")
    vsl_results = train_classifiers(X_train_vsl, X_test_vsl, y_train, y_test)

    # Combine results from classifiers
    combined_train_results = np.column_stack((
        mpqa_results['DecisionTree'][2], mpqa_results['SVM'][2], mpqa_results['RandomForest'][2],
        swn_results['DecisionTree'][2], swn_results['SVM'][2], swn_results['RandomForest'][2],
        vsl_results['DecisionTree'][2], vsl_results['SVM'][2], vsl_results['RandomForest'][2],
    ))

    # Train custom IBA-ENet classifier
    print("Training IBA-ENet classifier:")
    iba_enet = build_iba_enet(combined_train_results.shape[1])
    iba_enet.fit(combined_train_results, y_test, epochs=10, batch_size=32, verbose=1)

    # Evaluate custom classifier
    combined_test_results = np.column_stack((
        mpqa_results['DecisionTree'][2], mpqa_results['SVM'][2], mpqa_results['RandomForest'][2],
        swn_results['DecisionTree'][2], swn_results['SVM'][2], swn_results['RandomForest'][2],
        vsl_results['DecisionTree'][2], vsl_results['SVM'][2], vsl_results['RandomForest'][2],
    ))

    _, accuracy = iba_enet.evaluate(combined_test_results, y_test)
    print(f"IBA-ENet Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
