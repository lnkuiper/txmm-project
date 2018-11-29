from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
from nltk import pos_tag
import nltk
import os
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.svm import SVC
from sklearn.preprocessing import scale
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif, RFECV
import struct
import string
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings("ignore")


# Download the 'stopwords' and 'punkt' from the Natural Language Toolkit, you can comment the next lines if already present.
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))

dataset = '50'
test = False

# Load the dataset into memory from the filesystem
def load_data(dir_name):
    return sklearn.datasets.load_files('../data/%s' % dir_name, encoding='utf-8')


def load_train_data():
    return load_data(dataset + '/train/')


def load_test_data():
    return load_data(dataset + '/test/')


# Extract features from a given text
def extract_features(text, w2v_dict):
    bag_of_words = [x for x in wordpunct_tokenize(text)]
    sentences = [x for x in sent_tokenize(text)]

    features = []

    # Example feature 1: count the number of words
    features.append(len(bag_of_words))

    # Example feature 2: count the number of words, excluded the stopwords
    features.append(len([x for x in bag_of_words if x.lower() not in stop_words]))

    # Feature 3: count the number of sentences
    features.append(len(sentences))

    # Feature 4: average words per sentence
    features.append(len(bag_of_words)/len(sentences))

    # Features 5-9: count the number of digits, spaces, lowercase, uppercase and other characters
    features.append(sum(c.isdigit() for c in text))
    features.append(sum(c.isspace() for c in text))
    features.append(sum(c.islower() for c in text))
    features.append(sum(c.isupper() for c in text))
    features.append(len(text) - sum(features[4:8]))

    # Feature 10: each punctuation count
    punctuation = ['.',',','?','!','(',')',':',';','\'','\"', '<', '>', '\\', '/']
    features += [text.count(p) for p in punctuation]

    # Feature 11: counts of each upper- and lowercase letter
    features += [text.count(l) for l in list(string.ascii_lowercase) + list(string.ascii_uppercase)]

    # Feature 12: 'he' and 'she' frequency
    features.append(sum(w == 'he' for w in bag_of_words))
    features.append(sum(w == 'she' for w in bag_of_words))

    # Feature 13: amount of unique words
    features.append(len(set(bag_of_words)))

    # Feature 14: POS tag count
    all_tags = ['CC','CD','DT','EX','IN','JJ','JJR','JJS','LS','MD','NN','NNP','NNS','PDT','POS','PRP',
                'RB','RBR','RBS','RP','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WRB']
    tags = [p[1] for p in pos_tag(bag_of_words)]
    features += [tags.count(pos) for pos in all_tags]

    # Feature 15: count of each stopword
    features += [bag_of_words.count(sw) for sw in stop_words]

    # Feature 16: Emoticon count
    emoticons = [':-)',':)','=)',':-d',':d','=d','x)','xd',':-(',':(','=(',':\'-(',':\'(',';-)',
                ';)',':-p',':p','=p',';-p',';p','^^','>:-(','>:(','>:c',':-|',':|','-.-',':-s',
                ':s','=s','x.x',':-o',':o',':-o',':o','o.o','o_o','-.-','-,-','-.-"','-,-"','-_-']
    features += [text.lower().count(e) for e in emoticons]

    # Feature 17: total emoticon count 
    features.append(sum([any(c in w.lower() for c in emoticons) for w in bag_of_words]))

    # Feature 18: count of each specific digit
    features += [text.count(x) for x in [str(y) for y in range(10)]]

    # Feature 19: amount of words without vowels
    features.append(sum([any(c in w.lower() for c in r'aeiou') for w in bag_of_words]))

    # Feature 20: word vector average
    features += list(sum([w2v_dict.get(word) for word in bag_of_words if word in w2v_dict])/len(bag_of_words))

    return features


# Classify using the features
def classify(train_features, train_labels, test_features):
    # TODO: (Optional) If you would like to test different how classifiers would perform different, you can alter
    # TODO: the classifier here.
    # clf = MLPClassifier(hidden_layer_sizes=(256,)) # Performs significantly worse
    clf = SVC(kernel='linear')
    clf.fit(train_features, train_labels)
    return clf.predict(test_features)


# Evaluate predictions (y_pred) given the ground truth (y_true)
def evaluate(y_true, y_pred):
    # TODO: What is being evaluated here and what does it say about the performance? Include or change the evaluation
    # TODO: if necessary.
    recall = sklearn.metrics.recall_score(y_true, y_pred, average='macro')
    print("Recall: %f" % recall)

    precision = sklearn.metrics.precision_score(y_true, y_pred, average='macro')
    print("Precision: %f" % precision)

    f1_score = sklearn.metrics.f1_score(y_true, y_pred, average='macro')
    print("F1-score: %f" % f1_score)

    return recall, precision, f1_score

def bool_mask_list(list_a, mask):
    return [a for (a, b) in zip(list_a, mask) if b]

# The main program
def main():
    print('Loading train data...')
    train_data = load_train_data()
    print('Loading train data complete. Fitting Word2Vec model...')

    bags_of_words = [wordpunct_tokenize(text.lower()) for text in train_data.data]
    model = Word2Vec(bags_of_words, size=200)
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    print('Word2Vec model fit. Extracting features...')

    # Extract the features
    features = [extract_features(text, w2v) for text in tqdm(train_data.data)]
    print('\nNumber of features before selection:', len(features[0]))
    features = scale(features)
    estimator = SVC(kernel='linear')
    selector = RFECV(estimator, step=20, cv=5, scoring='f1_micro', verbose=1, n_jobs=-1)
    features = selector.fit_transform(features, train_data.target)
    mask = list(selector.support_)
    print('\nNumber of features after selection:', len(features[0]))
    # features = SelectKBest(mutual_info_classif, k=100).fit_transform(features, train_data.target)
    print('Extraction complete. Starting.')

    # Classify and evaluate
    skf = sklearn.model_selection.StratifiedKFold(n_splits=10)
    scores = []
    for fold_id, (train_indexes, validation_indexes) in enumerate(skf.split(train_data.filenames, train_data.target)):
        # Print the fold number
        print("Fold %d" % (fold_id + 1))

        # Collect the data for this train/validation split
        train_features = [features[x] for x in train_indexes]
        train_labels = [train_data.target[x] for x in train_indexes]
        validation_features = [features[x] for x in validation_indexes]
        validation_labels = [train_data.target[x] for x in validation_indexes]

        # Classify and add the scores to be able to average later
        y_pred = classify(train_features, train_labels, validation_features)
        scores.append(evaluate(validation_labels, y_pred))

        # Print a newline
        print("")

    # Print the averaged score
    recall = sum([x[0] for x in scores]) / len(scores)
    print("Averaged total recall", recall)
    precision = sum([x[1] for x in scores]) / len(scores)
    print("Averaged total precision", precision)
    f_score = sum([x[2] for x in scores]) / len(scores)
    print("Averaged total f-score", f_score)
    print("")

    # TODO: Once you are done crafting your features and tuning your model, also test on the test set and report your
    # TODO: findings. How does the score differ from the validation score? And why do you think this is?
    if test:
        print('Scores on test data:')
        test_data = load_test_data()
        test_features = [extract_features(text, w2v) for text in tqdm(test_data.data)]
        test_features = scale(test_features)
        test_features = [bool_mask_list(feats, mask) for feats in test_features]

        y_pred = classify(features, train_data.target, test_features)
        evaluate(test_data.target, y_pred)

# This piece of code is common practice in Python, is something like if "this file" is the main file to be ran, then
# execute this remaining piece of code. The advantage of this is that your main loop will not be executed when you
# import certain functions in this file in another file, which is useful in larger projects.
if __name__ == '__main__':
    main()