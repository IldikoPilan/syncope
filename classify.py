# -*- coding: utf-8 -*-
"""
@author: ILDPIL
"""

import os
import codecs
import csv
import pickle
import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, ParameterGrid
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from collections import Counter
from preprocess import has_mention, count_dept


###############
# Data loading
###############

def get_class_mapping(loaded_data, print_info=True):
    """ Returns a list of tuples with (label id, label name) for 
    the class labels (y) in the dataset.
    """
    label_mapping = [(i,label) for i,label in enumerate(loaded_data.target_names)]
    if print_info:
        print('Label mapping:')
        for target, name in label_mapping:
            print('\t{:<4} = {}'.format(target, name))
    print()
    return label_mapping

def get_dataset_info(loaded_data, y_train, y_val, y_test, label_mapping):
    """ Prints the distribution of instances per label 
    in the training and test sets. 
    """
    class0_train = [el for el in y_train if el == label_mapping[0][0]] # 0=_Pre_R55, 1=R55
    class0_val = [el for el in y_val if el == label_mapping[0][0]]
    class0_test = [el for el in y_test if el == label_mapping[0][0]]
    print('Total nr of instances: {}'.format(len(y_train)+len(y_val)+len(y_test)))
    print()
    print('{:<8}{:<8}{:<8}{:<8}'.format('Dset', 'Total', label_mapping[0][1],label_mapping[1][1]))
    print('{:<8}{:<8}{:<8}{:<8}'.format('Train', len(y_train), len(class0_train), len(y_train)-len(class0_train)))
    print('{:<8}{:<8}{:<8}{:<8}'.format('Val', len(y_val), len(class0_val), len(y_val)-len(class0_val)))
    print('{:<8}{:<8}{:<8}{:<8}'.format('Test', len(y_test), len(class0_test), len(y_test)-len(class0_test)))
    print()

def insert_ids(loaded_data):
    """ Inserts text id (filename) as the first token for each text.
    (Ids of digits only will be excluded from BOW model). 
    """
    X_with_id = [] 
    for ix, instance in enumerate(loaded_data["data"]):
        if '/' in loaded_data.filenames[ix]:                      # news20
            text_id = loaded_data.filenames[ix].split('/')[-1]
        else:
            text_id = loaded_data.filenames[ix].split('\\')[-1].split('.')[0]
        X_with_id.append(text_id + ' ' + instance)
    return X_with_id

def load_data(path_to_dataset, categories):
    """ Loads and splits data. Presupposes the following folder structure: 
    each document as a separate file, divided into separate folders per class. 
    """
    try:
        loaded_data = load_files(path_to_dataset, encoding='utf-8', 
                                 categories=categories) #shuffle=False
    except UnicodeDecodeError:
        loaded_data = load_files(path_to_dataset, encoding='latin1', 
                                 categories=categories)
    X = insert_ids(loaded_data)
    y = loaded_data["target"]
    return loaded_data, X, y


#####################
# Feature extraction
#####################

def dummy_tokenizer(text):
    """ Word boundary is white-space (since input is space-joined UDPipe-parsed text).  
    Word is any alphanumberic token at least two characters long with at least one 
    alphabetical character.
    """
    return [w for w in text.split() if w.isalnum() and not w.isdigit() and len(w) > 1]

def load_wordlist(path_to_list):
    """ Loads words from a file with one word per line into a list.
    """
    with codecs.open(path_to_list, 'r', 'utf-8') as f:
        words = [w.strip('\r\n') for w in f.readlines()] 
        stop_words = words + [w.strip('\r\n').title() for w in words]
    return stop_words

def extract_features(vect_type, X_train, stop_words, lowercase):
    """ Extracts bag-of-words (bow) features from the training data
    through vectorization. 
    """
    if vect_type == 'count':
        vectorizer = CountVectorizer(lowercase=lowercase)
    else:
        vectorizer = TfidfVectorizer(lowercase=lowercase, 
                                     stop_words=frozenset(stop_words), 
                                     tokenizer=dummy_tokenizer)
                                     # ngram_range=(1,3)
                                     # min_df=1, # minimum doc frequency (ignores terms appearing in <n docs)
    vectorizer.fit(X_train)
    bow = vectorizer.transform(X_train)
    print('Nr of words in vocab: {}'.format(len(vectorizer.vocabulary_)))
    print()
    return vectorizer, bow

def get_most_freq_words(bow, vectorizer, topn=100):
    """ Returns a list of the most frequent words in the training data sorted
    in decreasing order of frequency.
    """
    sum_words = bow.sum(axis=0)
    word_freqs = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    sorted_vocab = sorted(word_freqs, key= lambda x: x[1], reverse=True)
    for i, (w, count) in enumerate(sorted_vocab[:topn]):
        print('{:<5}{:<8.2f}{:<15}'.format(i+1, count, w))
    print()
    return sorted_vocab


###########
# Training
###########

def grid_search_train(estimator, bow, y_train, X_val, y_val):
    """ Performs a grid search over some hyperparameters for the estimator
    using cross-validation on the training data.
    """
    classifiers = {'lr':LogisticRegression(),
                   'svm':svm.SVC()}
    params = {'lr':{'penalty':['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100], 
                  'solver':['liblinear']},
              'svm':[{'kernel':['rbf'], 'gamma':[1e-3, 1e-4], 
                   'C': [0.01, 0.1, 1, 10, 100]},
                  {'kernel':['linear'], 'C': [0.01, 0.1, 1, 10, 100]}]}    
    best=(0, None, None)
    for grid in ParameterGrid(params[estimator]):
        classifier = classifiers[estimator].set_params(**grid)
        classifier.fit(bow, y_train)
        y_pred = classifier.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        if acc > best[0]:
            best = (round(acc,4), grid, classifier)
    best_acc, best_grid, best_classifier = best
    print('Best params: ', best_grid)
    print('Best val acc: ', best_acc)
    return best_classifier

def held_out_train(estimator, bow, y_train):
    """ Trains a classifier on a held-out set of training instances.
    """
    if estimator == 'lr':
        classifier = LogisticRegression(C=10, penalty = 'l1', solver = 'liblinear')
    else:
        classifier = svm.SVC(kernel='linear', C=1) 
    classifier.fit(bow, y_train)
    return classifier


#############
# Evaluation
#############

def plot_learning_curve(classifier, X, y):
    train_sizes = [100, 200, 300, 400, 500]
    train_sizes, train_scores, test_scores = learning_curve(classifier, X, y, 
                                                            train_sizes=train_sizes, cv=5)
    plt.figure()
    plt.title("Learning curves")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy score")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    plt.savefig('learning_curve.png', format='png')
    plt.show()

def plot_ROC(test_labels, test_predictions):
    fpr, tpr, thresholds = roc_curve(
        test_labels, test_predictions, pos_label=1) # R_55
    auc_res = "%.2f" % auc(fpr, tpr)
    title = 'ROC Curve, AUC = '+str(auc_res)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, "#000099", label='ROC curve')
        ax.plot([0, 1], [0, 1], 'k--', label='Baseline')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.title(title)
        plt.savefig('ROC.png', format='png')
        plt.show()

def plot_info_feats(coefs_with_fns):
    """ Produces a bar plot with the most informative
    features and their weight.
    """
    plt.rcParams.update({'font.size':11})
    imp = [el[0] for el in coefs_with_fns]
    names = [el[1] for el in coefs_with_fns]
    plt.figure(figsize=(10,7))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.savefig('mif_{}.png'.format(len(imp)), format='png')
    plt.show()

def get_most_informative_features(vectorizer, clf, n=20, out_format='list'):
    """ Prints the most informative features (words) in decreasing order 
    together with their weight.
    """
    feature_names = vectorizer.get_feature_names()
    try:
        coefs = clf.coef_.toarray()
    except AttributeError:
        coefs = clf.coef_
    coefs_with_fns = sorted(zip(coefs[0], feature_names), reverse=True)[:n]
    if out_format == 'plot':
        plot_info_feats(coefs_with_fns)
    else:
        for i, (coef, feat) in enumerate(coefs_with_fns):
            print('{:<4}{:<15}{:<10.3f}'.format(i+1, feat, coef))
    print()
    return coefs_with_fns

def get_misclassified(classifier, vectorizer, loaded_data, X_test, y_test, label_mapping, test_size, random_state=1, save_info=True):
    """ Returns a list of tuples with (text_id, text) for
    each misclassified instance in X_test. Values for 'test_size' 
    and 'random_state' should be the same as those for train_test_split.
    TO DO: debug, text ids don't match original
    """
    miscl_texts = []
    y_test = np.asarray(y_test)
    y_pred = classifier.predict(vectorizer.transform(X_test))
    misclassified = list(np.where(y_test != y_pred)[0]) # indices into X_test
    for ix in misclassified:
        text_id = X_test[ix].split()[0]
        mapped_label = label_mapping[int(y_pred[ix])][1]
        mapped_label_orig = label_mapping[int(y_test[ix])][1]
        miscl_texts.append((text_id, X_test[ix], mapped_label, mapped_label_orig))
    print('Misclassified instances:')
    print('{:<10}\t{:<20}\t{:<20}'.format('JournalID', 'Pred', 'Orig'))
    for text_id, text, mapped_label, mapped_label_orig in sorted(miscl_texts):
        print('{:<10}\t{:<20}\t{:<20}'.format(text_id, mapped_label, mapped_label_orig))
    if save_info:
        out_fn = 'misclassified.csv'
        with codecs.open(out_fn, 'w', 'utf-8') as f:
            csv_writer = csv.writer(f, delimiter='\t', dialect='excel-tab')
            for text_id, text, mapped_label, mapped_label_orig in sorted(miscl_texts):
                csv_writer.writerow(list((text_id, mapped_label, text)))
        print('Misclassified instances saved to {}.'.format(os.path.join(os.getcwd(), out_fn)))
        print()
    return miscl_texts

def analyze_misclassified(miscl_texts, labels, target_strings):
    """ Computes statistics for misclassified instances about
    target word mentions, and label and department distribution. 
    """
    class_cnt = Counter()
    mention_cnt = Counter()
    for text_id, text, label, mapped_label_orig in miscl_texts:
        class_cnt[label] += 1
        if has_mention(text, target_strings):
            mention_cnt[label+'-'+text_id] += 1
    print('Analysis of misclassified instances (for predicted labels):')
    print('Class count')
    for label, cnt in class_cnt.items():
        print('\t{:<10}{}'.format(label, cnt))
    print('Mention count for "{}":'.format( ", ".join(target_strings)))
    for label, cnt in mention_cnt.items():
        print('\t{:<10}{}'.format(label, cnt))

def baseline_classify(X_test, lowercase=False):
    """ Predict with keyword matching as baseline.
    """
    y_pred = []
    for text in X_test:
        if lowercase:
            text = text.lower()
        if 'synkop' in text:
            y_pred.append(1)
        else:
            y_pred.append(0)
    print('nr instances: ', len(y_pred))
    print('nr texts with keywords: ', len([el for el in y_pred if el==1]))
    print(y_pred)
    return np.array(y_pred)

########################
# Classification process 
########################

def do_classification(path_to_dataset, random_state, test_size, 
                      vect_type, lowercase, path_to_stoplist, categories,
                      train_type='all', estimator='svm',
                      model_name='', plot_lcurve=False, plot_ROC_curve=False,
                      show_inf_feat=False, get_miscl=False):
    """ Performs the complete process of classification from
    data loading to model evaluation.
    """
    print('''Classification params:
          vectorizer = {} 
          lowercase = {}
          train_type = {} 
          estimator = {}'''.format(vect_type, lowercase, train_type, estimator))
    print()
    # Get features
    loaded_data, X, y = load_data(path_to_dataset, categories)
    # shuffle (also by default) using 'random_state' as seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, 
                                                        test_size=test_size, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=random_state, 
                                                        test_size=len(y_test), stratify=y_train)
    label_mapping = get_class_mapping(loaded_data)
    get_dataset_info(loaded_data, y_train, y_val, y_test, label_mapping)
    
    if estimator == 'baseline':
        y_pred_val = baseline_classify(X_val, lowercase)
        print('Acc baseline - VAL: ', round(accuracy_score(y_val,y_pred_val),3))
        y_pred = baseline_classify(X_test, lowercase)
        print('Acc baseline - TEST: ', round(accuracy_score(y_test,y_pred),3))
    
    else:
        stop_words = load_wordlist(path_to_stoplist)
        vectorizer, bow = extract_features(vect_type, X_train, stop_words, lowercase)
        # Train
        if train_type == 'grid':  # (grid search & cross-validation)
            classifier = grid_search_train(estimator, bow, y_train, vectorizer.transform(X_val), y_val)
        else:                     # train on held-out set
            classifier = held_out_train(estimator, bow, y_train)
        if model_name:
            fn = '{}.model'.format(model_name)
            with open(fn, 'wb') as f:
                pickle.dump(classifier, f)
            print('Model saved to: {}'.format(os.path.join(os.getcwd(), fn)))
            # Loading saved model
            #with open(fn, 'rb') as f:   
            #    classifier = pickle.load(f)
        if plot_lcurve:
            plot_learning_curve(svm.SVC(kernel='linear', C=1), bow, y_train)                    
        if show_inf_feat:
            get_most_informative_features(vectorizer, classifier, n=30, out_format='plot') #list
        # Test & evaluate
        y_pred = classifier.predict(vectorizer.transform(X_test))
        if plot_ROC_curve:
            plot_ROC(y_test, y_pred)
        print()
        print('Classification report on test data')
        print(classification_report(y_test, y_pred, target_names = loaded_data.target_names))
        if get_miscl:
            miscl_texts = get_misclassified(classifier, vectorizer, loaded_data, X_test, 
                                            y_test, label_mapping, test_size, save_info=False)
            target_strings = ['synkop', 'syncop']
            analyze_misclassified(miscl_texts, loaded_data.target_names, target_strings)

