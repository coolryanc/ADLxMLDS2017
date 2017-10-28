import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense, LSTM, Dropout, Masking, BatchNormalization, Conv1D, GRU
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import load_model
from keras.callbacks import EarlyStopping

def readTrainingData():
    print('Parsing Training Data ...')
    record_frame = []
    frame_sequence = {}
    mfcc_Y = []
    mfcc_label_dic = {}
    fe_phone_char_dic = {}
    mfcc_train = pd.read_csv('./data/mfcc/train.ark', delim_whitespace=True, header=None).as_matrix() #shape: (1124823, 40)
    mfcc_label = pd.read_csv('./data/label/train.lab', header=None).as_matrix() #shape: (1124823, 2)
    fe_phone_char = pd.read_csv('./data/48phone_char.map', delim_whitespace=True, header=None).as_matrix() #shape: (48,3)
    # fbank_train = pd.read_csv('./data/fbank/train.ark', delim_whitespace=True, header=None).as_matrix() #shape: (1124823, 70)
    print('Build dict ...')
    for item in mfcc_label:
        mfcc_label_dic[item[0]] = item[1]
    for item in fe_phone_char:
        fe_phone_char_dic[item[0]] = item[1]
    # mfcc_train = np.concatenate((mfcc_train, fbank_train[:,1:]), axis=1)
    print(mfcc_train.shape)
    for index, item in enumerate(mfcc_train):
        print('\rIteration: {}, Find label for training data {}'.format(index, item[0]), end='', flush=True)
        frame_label = mfcc_label_dic[item[0]]
        splitFrameName = item[0].split('_')
        getFrameName = splitFrameName[0]+splitFrameName[1]
        if getFrameName not in frame_sequence:
            frame_sequence[getFrameName] = 0
            r_index = {"start": index, "count":0}
            record_frame.append(r_index)
        frame_sequence[getFrameName] += 1
        record_frame[-1]["count"] += 1
        mfcc_Y.append(fe_phone_char_dic[frame_label])
    print()
    mfcc_Y = np.array(mfcc_Y)
    mfcc_Y = np_utils.to_categorical(mfcc_Y, num_classes=49)
    mfcc_train, mfcc_Y = paddingData(mfcc_train[:,1:], mfcc_Y, frame_sequence, record_frame)
    return mfcc_train, mfcc_Y

def paddingData(mfcc_train, mfcc_Y, frame_sequence, record_frame): #mfcc_train shape: (1124823, 40)
    max_seq_len = max(frame_sequence.values()) # 777
    expand_train = np.zeros((max_seq_len*len(frame_sequence), mfcc_train.shape[1])) # init
    expand_Y = np.zeros((max_seq_len*len(frame_sequence), mfcc_Y.shape[1])) # init
    for index, item in enumerate(record_frame):
        numberOfRow = max_seq_len - item["count"]
        print('\rExpand at index {}'.format(item["start"]), end='', flush=True)
        insertIndex = item["start"]
        endIndex = item["start"] + item["count"]
        get_mfcc = mfcc_train[insertIndex:endIndex,:]
        fill = np.full((numberOfRow, mfcc_train.shape[1]), 0)
        expand_train[index*max_seq_len:index*max_seq_len+max_seq_len,:] = np.concatenate((get_mfcc,fill),axis=0)
        get_mfcc_Y = mfcc_Y[insertIndex:endIndex,:]
        fill_Y = np.full((numberOfRow, mfcc_Y.shape[1]), 0)
        fill_Y[:,-1] = 1
        expand_Y[index*max_seq_len:index*max_seq_len+max_seq_len,:] = np.concatenate((get_mfcc_Y,fill_Y),axis=0)
    return expand_train, expand_Y


if __name__ == '__main__':
    mfcc_train, mfcc_Y = readTrainingData() # trainingData & label
    # print(mfcc_train.shape)
    # np.save('./expandData/mfcc_fbank_train.npy', mfcc_train)
    # np.save('./expandData/mfcc_fbank_Y.npy', mfcc_Y)
    # print("Read training data and label ... ")
    # mfcc_train = np.load('./expandData/mfcc_train.npy')
    # mfcc_Y = np.load('./expandData/mfcc_Y.npy')

    mfcc_train = mfcc_train.reshape(-1,777,mfcc_train.shape[1])
    mfcc_Y = mfcc_Y.reshape(-1,777,49)

    mfcc_test = mfcc_train[int(len(mfcc_train)*9/10):]
    mfcc_Y_test = mfcc_Y[int(len(mfcc_Y)*9/10):]

    mfcc_train = mfcc_train[:int(len(mfcc_train)*9/10)]
    mfcc_Y = mfcc_Y[:int(len(mfcc_Y)*9/10)]

    LEARNING_RATE = 0.001
    OUTPUT_SIZE = mfcc_Y.shape[2] # 48phone_char
    ITERATION = 40
    BATCH_SIZE = 100
    CELL_SIZE = 256 # numbers of neural unit
    TIME_STEPS = 777 #

    # build RNN model
    model = Sequential()
    model.add(Masking(mask_value = 0, input_shape=(mfcc_train.shape[1], mfcc_train.shape[2])))
    model.add(BatchNormalization())
    model.add(LSTM(256, batch_input_shape=(None, mfcc_train.shape[1], mfcc_train.shape[2]), return_sequences = True, dropout=0.2, recurrent_dropout=0.2))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(256, return_sequences = True, dropout=0.2, recurrent_dropout=0.2)))
    model.add(BatchNormalization())
    model.add(Dense(units=CELL_SIZE, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(OUTPUT_SIZE))
    model.add(Activation('softmax'))

    optimizer = RMSprop()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.summary()
    # early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(mfcc_train, mfcc_Y, epochs=ITERATION, batch_size=BATCH_SIZE)

    score = model.evaluate(mfcc_test, mfcc_Y_test)
    print("Loss: {}".format(score[0]))
    print("Accuract: {}".format(score[1]))

    model.save('./model/model_rnn.h5')  # creates a HDF5 file 'my_model.h5'
    # del model
