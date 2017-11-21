import json
import numpy as np
import os
import re
import unicodedata
import random

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
    padding = {}
    vocabs_count = 0
    with open(dataPath) as jsonFile:
        label = json.loads(jsonFile.read())
        for index, item in enumerate(label):
            print("\rIteration: {}, read words in {}".format(index+1, item['id']), end='', flush=True)
            for s in item['caption']:
                words = normalizeString(s).split(' ')
                if len(words) < 30:
                    for w in words:
                        if w not in vocabs:
                            vocabs[w] = vocabs_count
                            vocabs_count += 1
        vocabs['EOS'] = vocabs_count
        vocabs['BOS'] = vocabs_count + 1
        print('\nTotal words: {}'.format(len(vocabs)))
        for index, item in enumerate(label):
            print("\rIteration: {}, padding words in {}".format(index+1, item['id']), end='', flush=True)
            padding[item['id']] = []
            for s in item['caption']:
                words = normalizeString(s).split(' ')
                if len(words) < 9:
                    words.insert(0,'BOS')
                    words.append('EOS')
                    one = np.zeros((10,len(vocabs)))
                    for w_index, w in enumerate(words):
                        one[w_index][vocabs[w]] = 1
                    padding[item['id']].append(one)
        print()
        return padding, vocabs

def readTraingFeature():
    training_Y_dic, vocabs = readLabel('./data/training_label.json')
    training_Y = []
    training_data = []
    for dirPath, dirNames, fileNames in os.walk("./data/training_data/feat/"):
        for index, f in enumerate(fileNames):
            print("\rIteration: {}, read {}".format(index+1, f), end='', flush=True)
            data = np.load(dirPath + f)
            random_caption_index = random.randint(0, len(training_Y_dic[f[:-4]])-1)
            training_Y.append(training_Y_dic[f[:-4]][random_caption_index])
            training_data.append(data)
    print("\nTransform to np array")
    training_data = np.array(training_data)
    training_Y = np.array(training_Y)
    return training_data, training_Y

if __name__ == '__main__':
    readTraingFeature()
