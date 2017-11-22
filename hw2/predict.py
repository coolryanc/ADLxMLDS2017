import numpy as np
import pandas as pd
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
import json
import re
import unicodedata
from keras.models import load_model

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r"", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def readLabel(dataPath):
    vocabs = {}
    vocabs_count = 0
    with open(dataPath) as jsonFile:
        label = json.loads(jsonFile.read())
        for index, item in enumerate(label):
            print("\rIteration: {}, read words in {}".format(index+1, item['id']), end='', flush=True)
            for s in item['caption']:
                words = normalizeString(s).split(' ')
                if len(words) < 9:
                    for w in words:
                        if w not in vocabs:
                            vocabs[vocabs_count] = w
                            vocabs_count += 1
        vocabs['EOS'] = vocabs_count
        vocabs['BOS'] = vocabs_count + 1
        print('\nTotal words: {}'.format(len(vocabs)))
        return vocabs

def readTraingFeature(dataPath):
    testing_data = []
    test_id = pd.read_csv(dataPath+'testing_id.txt', header=None).as_matrix()
    for i in test_id:
        print("\rRead {}".format(i[0]), end='', flush=True)
        data = np.load(dataPath+"testing_data/feat/" + i[0] + '.npy')
        testing_data.append(data)
    print("\nTransform to np array")
    testing_data = np.array(testing_data)
    return testing_data

def readPeerFeature(dataPath):
    peer_testing_data = []
    peer_id = pd.read_csv(dataPath+'peer_review_id.txt', header=None).as_matrix()
    for i in peer_id:
        print("\rRead {}".format(i[0]), end='', flush=True)
        data = np.load(dataPath+"peer_review/feat/" + i[0] + '.npy')
        peer_testing_data.append(data)
    print("\nTransform to np array")
    peer_testing_data = np.array(peer_testing_data)
    return peer_testing_data

def decode_sequence(input_seq, encoder_model, decoder_model, vocabs):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, 5916))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, 0] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    len_sen = 0
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sample_word = vocabs[sampled_token_index]
        decoded_sentence += sample_word + " "
        len_sen += 1
        # Exit condition: either hit max length
        # or find stop character.
        if (sample_word == 'EOS' or len_sen > 9):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, 5916))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

def main(argv):
    testing_data = readTraingFeature(argv[1])
    peer_data = readPeerFeature(argv[1])
    test_id = pd.read_csv(argv[1]+'testing_id.txt', header=None).as_matrix() #./data
    peer_id = pd.read_csv(argv[1]+'peer_review_id.txt', header=None).as_matrix()
    vocabs = readLabel(argv[1]+'training_label.json')
    encoder_model = load_model('./model/s2s_en_1.h5')
    decoder_model = load_model('./model/s2s_de_1.h5')
    writeText = ""
    for seq_index in range(len(testing_data)):
        input_seq = testing_data[seq_index: seq_index + 1]
        print(test_id[seq_index][0], end=': ')
        decoded_sentence = decode_sequence(input_seq, encoder_model, decoder_model, vocabs)
        print(decoded_sentence)
        writeText += test_id[seq_index][0] + "," + decoded_sentence.strip(' ') + ".\n"
    peer_writeText = ""
    for seq_index in range(len(peer_data)):
        input_seq = peer_data[seq_index: seq_index + 1]
        print('\r{}'.format(peer_id[seq_index][0]), end='', flush=True)
        decoded_sentence = decode_sequence(input_seq, encoder_model, decoder_model, vocabs)
        peer_writeText += peer_id[seq_index][0] + "," + decoded_sentence.strip(' ') + "\n"

    filename = argv[2]
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(writeText)
    peer_filename = argv[3]
    os.makedirs(os.path.dirname(peer_filename), exist_ok=True)
    with open(peer_filename, "w") as f:
        f.write(peer_writeText)

if __name__ == '__main__':
    main(sys.argv)
