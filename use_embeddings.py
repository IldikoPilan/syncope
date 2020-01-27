from gensim.models import KeyedVectors
from keras.layers import Embedding
import numpy as np

def get_embedding_weights(path_to_emb, word_index, emb_type='fasttext'):
    """ Returns an embeddings matrix of shape 
    (length of word_index, dimensions of embeddings).
    @ word_index: mapping of words from data to indices.
    """
    if emb_type == 'fasttext':
        try:
            word_vectors = KeyedVectors.load(path_to_emb, mmap='r', )
        except:
            word_vectors = KeyedVectors.load_word2vec_format(path_to_emb, binary=False, limit=500000)
    else:
        print('Only loading of fasttext models is implemented.')
    vocabulary = word_vectors.vocab.keys()  # sort?
    print('Found %s word vectors.' % len(vocabulary))

    EMBEDDING_DIM = len(word_vectors[list(vocabulary)[0]])
    print(EMBEDDING_DIM)
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    nr_ooe = 0
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = word_vectors[word]
        except KeyError:
            nr_ooe += 1
            pass
    print('Nr OOE: {} (out of {} words in index)'.format(nr_ooe, len(word_index)))
    return embedding_matrix

def get_emb_layer(vocab_size, embedding_dims, embedding_weights, trainable):
    """ Maps vocab indices to embedding_dims dimensions. Either random 
    initialization of weights or based on pre-trained embeddings.
    """
    if embedding_weights != []:
        return Embedding(input_dim=vocab_size,    
                         output_dim=embedding_dims,
                         weights=[embedding_weights],
                         trainable=trainable) 
    else:
        return Embedding(input_dim=vocab_size,    
                         output_dim=embedding_dims)
