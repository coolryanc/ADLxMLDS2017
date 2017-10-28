import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense, LSTM, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.models import load_model
from collections import OrderedDict
import itertools

def readTestingData():
    frame_sequence = {}
    record_frame = []
    print('Parsing Testing Data ...')
    mfcc_test = pd.read_csv('./data/mfcc/test.ark', delim_whitespace=True, header=None).as_matrix() #shape: (1124823, 40)
    # fbank_test = pd.read_csv('./data/fbank/test.ark', delim_whitespace=True, header=None).as_matrix()
    # mfcc_test = np.concatenate((mfcc_test, fbank_test[:,1:]), axis=1)
    # print(mfcc_test.shape)
    for index, item in enumerate(mfcc_test):
        print('\rIteration: {}, Get id for testing data at {}'.format(index, item[0]), end='', flush=True)
        splitFrameName = item[0].split('_')
        getFrameName = splitFrameName[0]+"_"+splitFrameName[1]
        if getFrameName not in frame_sequence:
            frame_sequence[getFrameName] = 0
            r_index = {"id": getFrameName, "count":0, "start": index}
            record_frame.append(r_index)
        record_frame[-1]["count"] += 1
        frame_sequence[getFrameName] += 1
    print()
    mfcc_test = paddingTestData(mfcc_test[:,1:], frame_sequence, record_frame)
    return mfcc_test, record_frame

def paddingTestData(mfcc_test, frame_sequence, record_frame): #mfcc_test
    max_seq_len = 777
    expand_test = np.zeros((max_seq_len*len(frame_sequence), mfcc_test.shape[1])) # init
    for index, item in enumerate(record_frame):
        numberOfRow = max_seq_len - item["count"]
        print('\rExpand at index {}'.format(item["start"]), end='', flush=True)
        insertIndex = item["start"]
        endIndex = item["start"] + item["count"]
        get_mfcc = mfcc_test[insertIndex:endIndex,:]
        fill = np.full((numberOfRow, mfcc_test.shape[1]), 0)
        expand_test[index*max_seq_len:index*max_seq_len+max_seq_len,:] = np.concatenate((get_mfcc,fill),axis=0)
    return expand_test

def getResultLabel(result):
    final_map = {}
    fe_tn_char_dic = {}
    fe_phone_char_dic = {}
    char_map = {}
    fe_phone_char = pd.read_csv('./data/48phone_char.map', delim_whitespace=True, header=None).as_matrix() #shape: (48,3)
    fe_tn_char = pd.read_csv('./data/phones/48_39.map', delim_whitespace=True, header=None).as_matrix() #shape: (48,3)
    print('\nBuild dict ... map phone seq')
    for item in fe_phone_char:
        fe_phone_char_dic[item[1]] = item[0]
        final_map[item[0]] = item[2]
    for item in fe_tn_char:
        fe_tn_char_dic[item[0]] = item[1]
    for key, value in fe_phone_char_dic.items():
        char_map[key] = final_map[fe_tn_char_dic[value]]
    concert_result = []
    for i in result:
        s = ""
        for j in i:
            origin_label = np.where(max(j)==j)[0][0]
            s += char_map[origin_label]
        concert_result.append(s)
    return concert_result

if __name__ == '__main__':
    mfcc_test, record_frame = readTestingData()
    mfcc_test = mfcc_test.reshape(-1,777,mfcc_test.shape[1])
    model = load_model('./model/model_mfcc_bid.h5')
    result = model.predict(mfcc_test, batch_size=500, verbose=0)
    writeText = "id,phone_sequence\n"
    result = getResultLabel(result[:,:,:-1])
    for index, item in enumerate(record_frame):
        s = result[index]
        s = ''.join(ch for ch, _ in itertools.groupby(s))
        s = s.strip("L")
        writeText += item["id"] + "," + s + '\n'
    with open('./result_10.csv', "w") as f:
        f.write(writeText)
