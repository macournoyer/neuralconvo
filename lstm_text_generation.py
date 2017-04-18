#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''(C) 2016 rust

Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.



'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys


#pandas  安装出错，所以直接从csv中解析
text_lines = open('000001.csv').readlines()[1:]
print ('day count = ' + str(len(text_lines)))

print (text_lines[0])

sources_all = []
targets_all = []

for line in reversed(text_lines):
    #print(line)
    lw = line.split(',')
    S = [float(lw[3]), float(lw[5])/10000, float(lw[6])/10000]
    if len(sources_all) == 0:
        S.append(0.00001)
        S.append(0.00001)
        S.append(0.00001)

    else:
        last = sources_all[-1]
        S.append((S[0] - last[0])/last[0])
        S.append((S[1] - last[1])/last[1])
        S.append((S[2] - last[2])/last[2])


    sources_all.append(S)

    T = S[3]
    targets_all.append(T)


print(len(sources_all))
print(len(targets_all))


sources = sources_all[:5000]
targets = targets_all[:5000]
sources_test = sources_all[5000:]
targets_test = targets_all[5000:]


'''
path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
'''

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 1
sentences = []
next_chars = []
for i in range(0, len(sources) - maxlen, step):
    sentences.append(sources[i: i + maxlen])
    next_chars.append(targets[i + maxlen])
print('nb sequences:', len(sentences))

sentences_test = []
next_chars_test = []
for i in range(0, len(sources_test) - maxlen, step):
    sentences_test.append(sources_test[i: i + maxlen])
    next_chars_test.append(targets_test[i + maxlen])
print('nb test sequences:', len(sentences_test))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, 6), dtype=np.float32)
y = np.zeros((len(sentences), 1), dtype=np.float32)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        for g in xrange(0,6):
            X[i, t, g] = char[g]
    y[i, 0] = next_chars[i]


# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(1024, return_sequences=True, input_shape=(maxlen, 6)))
model.add(LSTM(1024, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
#model.add(Dense(1))
#model.add(Activation('softmax'))
model.add(Activation('linear'))

model.compile(loss='mse', optimizer='rmsprop')

'''
def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))
'''

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)


    predret = []
    for sent,targ in zip(sentences_test[:20], targets_test[:20]):
            x = np.zeros((1, maxlen, 6))
            for t, char in enumerate(sent):
                for g in xrange(0,6):
                    x[0, t, g] = char[g]

            preds = model.predict(x, verbose=0)[0]
            print(preds[0], targ)

    

print("cbf done!")
