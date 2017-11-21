import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Input, Activation, Dense, LSTM, Dropout, Masking, BatchNormalization, RepeatVector, GRU
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Model
from keras.callbacks import EarlyStopping
import sys
import os
import parseData
import random

if __name__ == '__main__':
    EPOCH = 100
    BATCH_SIZE = 128
    BATCH_INDEX = 0
    TIME_STEP = 80      # 80 frame
    INPUT_SIZE = 4096   # each frame has 4096 features
    LR = 0.01           # learning rate
    CELL_SIZE = 512
    encoder_input_data, decoder_input_data = parseData.readTraingFeature()
    decoder_inputs_dim = decoder_input_data.shape[2]

    decoder_target_data = np.zeros(
        (decoder_input_data.shape[0], decoder_input_data.shape[1], decoder_input_data.shape[2]))

    for i, caption in enumerate(decoder_input_data):
        for t, word in enumerate(caption):
            if t < len(caption)-1:
                decoder_target_data[i][t][:] = decoder_input_data[i][t+1][:]

    encoder_inputs = Input(shape=(None, 4096))
    encoder = LSTM(CELL_SIZE, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, decoder_inputs_dim))
    decoder_lstm = LSTM(CELL_SIZE, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(decoder_inputs_dim, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.summary()
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=BATCH_SIZE,
              epochs=EPOCH)
    # Save model
    model.save('./model/s2s.h5')

    #-----------------------------------------------------------------------
    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.save('./model/s2s_en_1.h5')

    decoder_state_input_h = Input(shape=(CELL_SIZE,))
    decoder_state_input_c = Input(shape=(CELL_SIZE,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    decoder_model.summary()
    decoder_model.save('./model/s2s_de_1.h5')
