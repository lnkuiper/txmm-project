from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
from nltk import pos_tag
import nltk
import os
import pickle
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
from math import log
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Word2Vec


# Download the 'stopwords' and 'punkt' from the Natural Language Toolkit, you can comment the next lines if already present.
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english'))

# Load the dataset into memory from the filesystem
def load_data(dir_name):
    return sklearn.datasets.load_files('../data/raw/%s' % dir_name, encoding='utf-8')


def load_train_data(dataset):
    return load_data(dataset + '/train/')


def load_test_data(dataset):
    return load_data(dataset + '/test/')


def idf(doc_freq, corpus_size):
    return log(corpus_size/(doc_freq+1))


# Extract features from a given text
def extract_features(text, w2v_dict, idf_dict={}):
    bag_of_words = [x.lower() for x in wordpunct_tokenize(text)]
    sentences = [x.lower() for x in sent_tokenize(text)]

    features = []

    # Example feature 1: count the number of words
    features.append(len(bag_of_words))

    # Example feature 2: count the number of words, excluded the stopwords
    features.append(len([x for x in bag_of_words if x.lower() not in stop_words]))

    # Feature 3: count the number of sentences
    features.append(len(sentences))

    # Feature 4: average words per sentence
    features.append(len(bag_of_words)/len(sentences))

    # Features 5-9: count the number of digits, lowercase, uppercase and other characters
    features.append(sum(c.isdigit() for c in text))
    features.append(sum(c.islower() for c in text))
    features.append(sum(c.isupper() for c in text))
    features.append(len(text) - sum(features[4:7]))

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
    for t1 in all_tags:
        features.append(tags.count(t1))
        for t2 in all_tags:
            features.append(sum(1 for i in range(len(all_tags)) if all_tags[i:i+2]==[t1, t2]))

    # Feature 15: count of each stopword
    features += [bag_of_words.count(sw) for sw in stop_words]

    # Feature 16: Emoticon count
    emoticons = [':-)',':)','=)',':-d',':d','=d','x)','xd',':-(',':(','=(',':\'-(',':\'(',';-)',
                ';)',':-p',':p','=p',';-p',';p','^^','>:-(','>:(','>:c',':-|',':|','-.-',':-s',
                ':s','=s','x.x',':-o',':o',':-o',':o','o.o','o_o','-.-','-,-','-.-"','-,-"','-_-']
    features += [text.lower().count(e) for e in emoticons]

    # Feature 17: total emoticon count 
    features.append(sum([any(c in w for c in emoticons) for w in bag_of_words]))

    # Feature 18: count of each specific digit
    features += [text.count(x) for x in [str(y) for y in range(10)]]

    # Feature 19: amount of words without vowels
    features.append(sum([any(c in w for c in r'aeiou') for w in bag_of_words]))

    # Feature 20: word vector average
    features += list(sum([w2v_dict.get(word)*idf_dict.get(word, 1) for word in bag_of_words if word in w2v_dict])/len(bag_of_words))

    # Feature 21: bigrams
    tokens = string.ascii_lowercase + '.,?!():;\'\"<>\\/- '
    for c1 in tokens:
        features.append(text.lower().count(c1))
        for c2 in tokens:
            features.append(text.lower().count(c1 + c2))

    return features


# Classify using the features
def classify(train_features, train_labels, test_features):
    # TODO: (Optional) If you would like to test different how classifiers would perform different, you can alter
    # TODO: the classifier here.
    # clf = MLPClassifier(hidden_layer_sizes=(256,)) # Performs significantly worse
    clf = SVC(kernel='linear', probability=True)
    clf.fit(train_features, train_labels)
    return clf.predict(test_features), clf.predict_log_proba(test_features)


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
def experiments(dataset='50', test=False):
    print('Dataset size:', dataset)
    # print('Loading train data...')
    train_data = load_train_data(dataset)
    # print('Loading data complete. Fitting Word2Vec model...')
    bags_of_words = [wordpunct_tokenize(text.lower()) for text in train_data.data]
    model = Word2Vec(bags_of_words, size=150)
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    idf_dict = {}
    # print('Word2Vec model fit. Computing IDFs...')
    words = [word for word in w2v]
    idfs = []
    # for word in tqdm(words):
    #     idfs.append(idf(sum([word in bag for bag in bags_of_words]), len(bags_of_words)))
    # idf_dict = dict(zip(words, idfs))
    # print('IDFs computed. Extracting features...')

    # Extract the features
    # print('Word2Vec model fit. Extracting features...')
    features = [extract_features(text, w2v, idf_dict) for text in train_data.data]
    # print('Number of features before selection: ' + str(len(features[0])) + '. Finding best set using recursive strategy...')
    features = scale(features)
    # estimator = SVC(kernel='linear')
    # selector = RFECV(estimator, step=3, cv=10, scoring='f1_micro', verbose=1, n_jobs=-1)
    # features = selector.fit_transform(features, train_data.target)
    # mask = list(selector.support_)
    with open('../optimal_feat_mask.pkl', 'rb') as f:
        mask = pickle.load(f)
    features = [bool_mask_list(feats, mask) for feats in features]
    # print('Number of features found: ' + str(len(features[0])) + '. Training SVC...')

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
        y_pred, y_pred_probas = classify(train_features, train_labels, validation_features)
        scores.append(evaluate(validation_labels, y_pred))

        # Save prediction probabilities
        proba_save_path = '../data/preds/' + dataset + '/train/'
        if not os.path.exists(proba_save_path):
            os.makedirs(proba_save_path)
        with open(proba_save_path + 'preds_split' + str(fold_id), 'wb+') as f:
            pickle.dump(y_pred_probas, f)
        with open(proba_save_path + 'labels_split' + str(fold_id), 'wb+') as f:
            pickle.dump(validation_labels, f)


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
        test_data = load_test_data(dataset)
        test_features = [extract_features(text, w2v, idf_dict) for text in test_data.data]
        test_features = scale(test_features)
        test_features = [bool_mask_list(feats, mask) for feats in test_features]

        y_pred, y_pred_probas = classify(features, train_data.target, test_features)
        evaluate(test_data.target, y_pred)

        proba_save_path = '../data/preds/' + dataset + '/test/'
        if not os.path.exists(proba_save_path):
            os.makedirs(proba_save_path)
        with open(proba_save_path + 'preds' + str(fold_id), 'wb+') as f:
            pickle.dump(y_pred_probas, f)
        with open(proba_save_path + 'labels' + str(fold_id), 'wb+') as f:
            pickle.dump(test_data.target, f)


def main():
    experiments(dataset=str(250), test=True)


# This piece of code is common practice in Python, is something like if "this file" is the main file to be ran, then
# execute this remaining piece of code. The advantage of this is that your main loop will not be executed when you
# import certain functions in this file in another file, which is useful in larger projects.
if __name__ == '__main__':
    main()
