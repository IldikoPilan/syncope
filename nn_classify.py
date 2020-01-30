# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:49:46 2019
@author: ILDPIL
"""
import os
import numpy as np
from numpy.random import seed
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import use_embeddings
import nn_models
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
#from keras import callbacks

def prepare_data(data_folder, lower, categories):
    # Load the UD-piped (tokenized) data
    try:
        syncope_data = load_files(data_folder, encoding='utf-8', categories = categories)
    except UnicodeDecodeError:
        syncope_data = load_files(data_folder, categories = categories) #, encoding='latin1'
    # TO DO: add text ID to texts
    X = [str(d) for d in syncope_data["data"]]
    y = syncope_data["target"]
    tokenizer = Tokenizer(lower=lower, split=' ') # filters default: punctuation, tabs, line breaks, but keeps '
    tokenizer.fit_on_texts(X)                     # Updates internal vocabulary based on a list of texts
    embeded_X = tokenizer.texts_to_sequences(X)  
    max_length = max([len(d) for d in embeded_X])
    # pad all docs to the same (max) length (pre-requisite in tensorflow)
    padded_X = pad_sequences(embeded_X, maxlen=max_length, padding='post') 
    return syncope_data, padded_X, y, tokenizer, max_length

def build_model(model_type, vocab_size, max_length, embedding_dims, embedding_weights, trainable):
    """ Set up different neural network architectures. 
    """
    print('Build "{}" model'.format(model_type))
    if model_type == 'joulin': 
        model = nn_models.joulin(vocab_size, embedding_dims, max_length)
    elif model_type == 'cnn_kim':
        model = nn_models.cnn_kim(vocab_size, max_length, embedding_dims, embedding_weights, trainable)
    elif model_type == 'lstm':
        model = nn_models.lstm(vocab_size, embedding_dims, embedding_weights, trainable)
    else:
        raise NameError('''No such model. Choose between:
                        "joulin", "cnn_kim" or "lstm"''')
    return model 

def plot_graphs(history, string):
    # Source: https://www.tensorflow.org/tutorials/text/text_classification_rnn
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string], '')
    plt.yticks(np.arange(0.5,1,0.05))
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

def evaluate(model, X_test, y_test, history, plot_history=True, class_report=False):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print('Testing accuracy: {:.4f} ({:.4f} loss)'.format(accuracy, loss))
    if plot_history:
        if 'ildikop' in os.getcwd():
            plot_graphs(history, 'accuracy')
        else:
            plot_graphs(history, 'acc') #accuracy in UiO Mac virtualenv
    if class_report:
        y_pred = model.predict(X_test, batch_size=64, verbose=0) # predicts probs       
        # Return the indices of the maximum values along an axis (1/-1 for horizontal)
        y_pred_1d_bool = y_pred.argmax(axis=-1) 
        print(classification_report(y_test, y_pred_1d_bool)) 
        
def do_nn_classification(data_folder, model_type, embedding_dims, path_to_emb, trainable,
                         epochs, lower, categories, out_model=''):
    random_state = 1
    print('categories: ', categories)
    print('lower: ', lower)
    print('emb dimensions: ', embedding_dims)
    print('epochs: ', epochs)
    loaded_data, padded_X, y, tokenizer, max_length = prepare_data(data_folder, lower, categories)
    X_train, X_test, y_train, y_test = train_test_split(padded_X, y, random_state=random_state, 
                                                      test_size=0.15, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=random_state, 
                                                      test_size=len(y_test), stratify=y_train)

    # For classification (with softmax, m units on last dense layer)
    y_train = to_categorical(y_train)   # convert to shape (N,m) N = nr of instances, m = nr classes
    y_test = to_categorical(y_test)     
    y_val = to_categorical(y_val) 
    
    print('Train','\t', 'Val','\t','Test')
    print(len(y_train),'\t', len(y_val),'\t', len(y_test))

    if path_to_emb:
        embedding_weights = use_embeddings.get_embedding_weights(path_to_emb, tokenizer.word_index)
        print('Using embeddings from:', path_to_emb)
        print('trainable: ', trainable)
    else:
        embedding_weights = []
    vocab_size = len(tokenizer.word_index) + 1
    model = build_model(model_type, vocab_size, max_length, embedding_dims,
                        embedding_weights, trainable)

    # Configure the learning process
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',             
                  metrics=['accuracy'])
    model.summary()
    
    # Early stopping - Stop training before overfitting
    #early_stop = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1) # requires validation_split()
    
    # Train
    history = model.fit(X_train, y_train,
              batch_size=50,
              epochs=epochs, 
              verbose=True,
              validation_data=(X_val, y_val))#,
              #callbacks=[early_stop])
    if out_model:
        model.save(out_model)
    evaluate(model, X_test, y_test, history) #plot_history=True, class_report=True
