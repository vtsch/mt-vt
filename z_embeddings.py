

import umap.umap_ as umap
import numpy as np
from keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed, Bidirectional
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential

def umap_embedding(data):
    reducer = umap.UMAP(n_components=2)
    umap_embedding = reducer.fit_transform(data)
    print("umap embedding shape: ", umap_embedding.shape)
    return umap_embedding

def lstm_embedding(timesteps, n_features):
    model = Sequential()
    model.add(LSTM(10, activation='relu', input_shape=(timesteps,n_features), return_sequences=True))
    model.add(LSTM(5, activation='relu', return_sequences=True))
    model.add(LSTM(10, activation='relu', return_sequences=True)) #each LSTM unit returning a sequence of 187 outputs,
    model.add(TimeDistributed(Dense(n_features))) # need to configure last LSTM layer prior to  TimeDistributed wrapped Dense layer to return sequences
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    return model

def simple_autoencoder():
    # size of encoded representation
    encoding_dim = 10
    #input sequence
    input = Input(shape=(187,))
    # encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input)
    # lossy reconstruction of the input
    decoded = Dense(187, activation='sigmoid')(encoded)
    model = Model(input, decoded)
    # This model maps an input to its encoded representation
    encoder_ac = Model(input, encoded)
    # This is our encoded input
    encoded_input_ac = Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer_ac = model.layers[-1]
    # Create the decoder model
    decoder_ac = Model(encoded_input_ac, decoder_layer_ac(encoded_input_ac))

    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    model.summary()  #print(autoencoder.summary())

    return model, encoded, encoder_ac

def deep_autoencoder():
    input = Input(shape=(187,))

    encoded = Dense(128, activation='relu')(input)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)

    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(187, activation='sigmoid')(decoded)

    model = Model(input, decoded)

    # This model maps an input to its encoded representation
    encoder_ac_deep = Model(input, encoded)
    encoded_input_ac_deep = Input(shape=(187,))

    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()

    return model, encoded, encoder_ac_deep

def lstm_autoencoder(timesteps, n_features):
    model = Sequential()
    # Encoder
    model.add(LSTM(30, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
    model.add(LSTM(10, activation='relu', return_sequences=False))
    model.add(RepeatVector(timesteps))
    # Decoder
    model.add(LSTM(10, activation='relu', return_sequences=True))
    model.add(LSTM(30, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    model.add(LSTM(187, return_sequences=False))

    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    model.summary()

    return model

def bilstm_autoencoder(timesteps):
    model = Sequential()
    model = Bidirectional(model)
    model.add(Embedding(1000, timesteps, input_length=187))
    model.add(LSTM(187))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()


