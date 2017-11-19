import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense, LSTM, Dropout, Masking, BatchNormalization, RepeatVector, GRU
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import load_model
from keras.callbacks import EarlyStopping
import sys
import os
import parseData
import random


if __name__ == '__main__':
    EPOCH = 10
    BATCH_SIZE = 50
    BATCH_INDEX = 0
    TIME_STEP = 80      # 80 frame
    INPUT_SIZE = 4096   # each frame has 4096 features
    LR = 0.01           # learning rate
    CELL_SIZE = 64
    x_train, y_train = parseData.readTraingFeature()
    # print(training_data.shape)
    # print(training_Y.shape) # (1450, captionNums, 15, 5260) ramdom get one caption

    model = Sequential()
    model.add(LSTM(batch_input_shape=(50, 80, 4096), output_dim=CELL_SIZE, return_sequences=False))
    model.add(Dense(15, activation="relu"))
    model.add(RepeatVector(15))
    model.add(LSTM(15, return_sequences=True))
    model.add(TimeDistributed(Dense(output_dim=15, activation="softmax")))
    optimizer = RMSprop()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()

    for step in range(EPOCH):
        print(step)
        # data shape = (batch_num, steps, inputs/outputs)
        X_batch = x_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
        Y_batch = []
        for c in y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE]:
            random_caption = random.randint(0, len(c)-1)
            Y_batch.append(np.array(c[random_caption]))
        # cost = model.train_on_batch(X_batch, Y_batch)
        # BATCH_INDEX += BATCH_SIZE
        # BATCH_INDEX = 0 if BATCH_INDEX >= x_train.shape[0] else BATCH_INDEX
        # if step % 500 == 0:
        #     cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
        #     print('test cost: ', cost, 'test accuracy: ', accuracy)
