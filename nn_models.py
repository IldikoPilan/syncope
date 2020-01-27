from keras.models import Sequential, Model
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Concatenate, Activation
from keras.layers import Conv1D, Convolution1D, MaxPooling1D, GlobalMaxPooling1D, Embedding, Dropout, Bidirectional, LSTM
from use_embeddings import *


def joulin(vocab_size, embedding_dims, max_length):
    """ Model based on Joulin et al. (https://arxiv.org/pdf/1607.01759.pdf)
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                        output_dim=embedding_dims,
                        input_length=max_length))                 
    model.add(GlobalAveragePooling1D()) # GlobalMaxPooling1D
    model.add(Dense(1, activation='sigmoid'))
    return model

def cnn_kim(vocab_size, max_length, embedding_dims, embedding_weights, trainable):
    # Partly based on: https://github.com/Tixierae/deep_learning_NLP/blob/master/cnn_imdb.ipynb
    nb_filters = 100        # 100 in Kim (2014)
    filter_sizes = [3,4,5]  # 3,4,5 in Kim (2014)
    drop_rate = 0.5         # 0.5 in Kim (2014)
    print('nr filters: ', nb_filters)
    print('filter sizes: ', filter_sizes) 
    print('dropout: ', drop_rate)
    input_layer = Input(shape=(max_length,)) 
    embedding_layer = get_emb_layer(vocab_size, embedding_dims, embedding_weights, trainable)
    embedded_sequences = embedding_layer(input_layer)
    embedded_sequences = Dropout(0.5)(embedded_sequences) 
    branches = []
    for filter_size in filter_sizes:
      conv = Conv1D(filters = nb_filters,
                    kernel_size = filter_size,
                    activation = 'relu')(embedded_sequences)
      pooled_conv = GlobalMaxPooling1D()(conv)
      pooled_conv = Dropout(drop_rate)(pooled_conv)
      branches.append(pooled_conv)
    concat = Concatenate()(branches)
    concat_dropped = Dropout(drop_rate)(concat)
    prob = Dense(units = 2,
                 activation = 'softmax')(concat_dropped)
    model = Model(input_layer, prob)
    return model

def lstm(vocab_size, embedding_dims, embedding_weights, trainable):
    model = Sequential()
    model.add(get_emb_layer(vocab_size, embedding_dims, embedding_weights, trainable))
    model.add(Bidirectional(LSTM(embedding_dims)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

