'''
TaikoNation v1.0
By Emily Halina & Matthew Guzdial
'''

import tflearn
import tensorflow as tf
import numpy as np
import os

def preprocess():
    '''
    this function preprocesses the dataset for use in the model

    1. load appropriate files
    2. slice input into needed chunks then flatten
    3. add these to train/test lists as specified
    '''

    charts = os.listdir(path="input_charts")
    songs = os.listdir(path="input_songs")
    fail_count = 0

    trainX = []
    trainY = []
    testX = []
    testY = []

    test_data = [2, 5, 9, 82, 28, 22, 81, 43, 96, 97]
    num = 0
    for chart in charts:
        # split the testing / training data
        training = True
        for h in test_data:
            if num == h:
                print("test_data")
                training = False
        
        # locate the matching song
        id_number = chart.split("_")[0]
        song = None
        for s in songs:
            if s.split()[0] == id_number:
                song = s
                break
        
        if song == None:
            raise FileNotFoundError
        else:
            print(song, chart)

        # load the song data from memory map and reshape appropriately        
        song_mm = np.load("input_songs/" + song, mmap_mode="r")
        song_data = np.frombuffer(buffer=song_mm, dtype=np.float32, count=-1)
        song_data = song_data[0:song_mm.shape[0]*song_mm.shape[1]]
        song_data = np.reshape(song_data, song_mm.shape)

        # load the note data as above
        if song_mm[0][0] != song_data[0][0]:
            fail_count += 1
        note_mm = np.load("input_charts_nr/" + chart, mmap_mode="r")
        note_data = np.frombuffer(note_mm, dtype=np.int32, count=-1)
        note_data = np.reshape(note_data, [len(note_mm), 7])

        # pad the note data with 0's so it matches the end of the song, adding exactly enough to make life easier for myself below
        diff = len(song_data) - len(note_data)
        padding = []
        for i in range(diff + 16):
            padding.append(np.zeros(7))

        note_data = np.append(note_data, padding, axis=0)

        # package up the last 16 blocks of data, which require padding in the song_input because the song ends
        for h in range(16):
            song_input = []
            note_input = []
            output_chunk = []
            for k in range(16):
                if h - k < 0:
                    song_input.append(np.zeros([80]))
                    if k < 12:
                        note_input.append(np.zeros([7]))
                    elif k != 15:
                        note_input.append(np.ones([7]))
                    if k > 11:
                        output_chunk.append(np.zeros([7]))
                else:
                    song_input.append(song_data[h-k])
                    if k < 12:
                        note_input.append(note_data[h-k])
                    elif k != 15:
                        note_input.append(np.ones([7]))
                    if k > 11: 
                        output_chunk.append(note_data[h-k])
            song_input = np.array(song_input).flatten()
            note_input = np.array(note_input).flatten()
            input_chunk = np.concatenate([song_input, note_input])
            output_chunk = np.concatenate(output_chunk)
            output_chunk = np.reshape(output_chunk, [4, 7])

            if training:
                trainX.append(input_chunk)
                trainY.append(output_chunk)
            else:
                testX.append(input_chunk)
                testY.append(output_chunk)
        
        # package up the data in 1445-size 1D tensors as needed for input, then append to appropriate Train / Test list
        for j in range(16, len(song_data)):
            song_input = []
            note_input = []
            output_chunk = []
            for k in range(16):
                song_input.append(song_data[j-k])
                if k < 12:
                    note_input.append(note_data[j-k])
                elif k != 15:
                    note_input.append(np.ones([7]))
                if k > 11:
                    output_chunk.append(note_data[j-k])
    
            song_input = np.array(song_input).flatten()
            note_input = np.array(note_input).flatten()
            input_chunk = np.concatenate([song_input, note_input])
            output_chunk = np.concatenate(output_chunk)
            output_chunk = np.reshape(output_chunk, [4, 7])
            if training:
                trainX.append(input_chunk)
                trainY.append(output_chunk)
            else:
                testX.append(input_chunk)
                testY.append(output_chunk)

        num += 1

    # ensure things are working
    print(len(trainX), "train X", len(trainY), "train Y")
    print(len(trainX[0]), "trainX 0", len(testX[0]), "testX 0")
    print(len(testX), "test X", len(testY), "test Y")
    print(len(trainY[0]), "trainY 0", len(testY[0]), "testY 0")

    return trainX,trainY,testX,testY

def main():
    '''
    Initial Architecture:
    Input: size 1385, [16, 80] and [7, 15] flattened and concatenated

    after unpacking the input, we put the song data through the following layers

    conv - 16 filters, filter_size 3, relu
        dropout - 80%
        max_pool - kernel 2

    conv - 32 filters, filter_size 3, relu
        max_pool - kernel 2

    fully connected - 128 nodes, relu

    from here, we bring in the previous data and do some multiplication to reshape
    the data to [16, 88] for input into the following layers

    lstm - 64 nodes, dropout 80%, relu
        reshape to [8,8]

    lstm - 64 nodes, dropout 80%, relu

    fully connected - 44 nodes, softmax (output layer)
    '''
    # commenting this and the last 2 lines out should let you run this script without the dataset
    trainX, trainY, testX, testY = preprocess()
    
    # unpack the input data
    net = tflearn.input_data([None, 1385])
    song = tf.slice(net, [0,0], [-1, 1280])
    song = tf.reshape(song, [-1, 16, 80])
    prev_notes = tf.slice(net, [0,1280], [-1, 105])
    prev_notes = tf.reshape(prev_notes, [-1, 7, 15])

    # two conv layers with a 20% dropout layer after the first and max_pooling after each
    song_encoder = tflearn.conv_1d(song, nb_filter=16, filter_size=3, activation="relu")
    song_encoder = tflearn.dropout(song_encoder, keep_prob=0.8)
    song_encoder = tflearn.max_pool_1d(song_encoder, kernel_size=2)
    print(song_encoder.shape, "song_encoder shape after conv 1")

    song_encoder = tflearn.conv_1d(song, nb_filter=32, filter_size=3, activation="relu")
    song_encoder = tflearn.max_pool_1d(song_encoder, kernel_size=2)
    print(song_encoder.shape, "song_encoder shape after conv 2")

    song_encoder = tflearn.fully_connected(song_encoder, n_units=128, activation="relu")
    print(song_encoder.shape, "song_encoder shape after fc")
    song_encoder = tf.reshape(song_encoder, [-1,8,16])

    # split song data into past chunks and current chunk
    past_chunks = tf.slice(song_encoder, [0,0,0], [-1, 8, 15])
    curr_chunk = tf.slice(song_encoder, [0,0,15], [-1, 8, 1])

    # combine the note data with the processed song data
    lstm_input = tf.unstack(past_chunks, axis=1)
    lstm_input = tf.math.multiply(lstm_input, prev_notes)
    lstm_input = tf.reshape(lstm_input, [-1]) # flatten this to add on the current chunk
    
    # add on the final segment which does not have data yet
    curr_chunk = tf.math.multiply(curr_chunk, tf.ones([8, 15]))
    curr_chunk = tf.reshape(curr_chunk, [-1])
    lstm_input = tf.concat([lstm_input, curr_chunk], 0)

    lstm_input = tf.reshape(lstm_input, [-1, 16, 88]) # reshape to desired shape
    print(lstm_input.shape, "shape of lstm input")
    
    # 2 lstm layers, then a final fully connected softmax layer
    lstm_input = tflearn.lstm(lstm_input, 64, dropout=0.8, activation="relu")
    lstm_input = tf.reshape(lstm_input, [-1, 8, 8])
    print(lstm_input.shape, "lstm_input shape after lstm 1 + reshape to 8 by 8 (64)")

    lstm_input = tflearn.lstm(song_encoder, 64, dropout=0.8, activation="relu")
    print(lstm_input.shape, "lstm_input shape after lstm 2")

    lstm_input = tflearn.fully_connected(lstm_input, n_units=28, activation="softmax")
    lstm_input = tflearn.reshape(lstm_input, [-1,4,7])
    print(lstm_input.shape, "lstm_input shape after final fc softmax layer")

    # setting up final parameters and running the fit
    network = tflearn.regression(lstm_input, optimizer = "adam", loss="categorical_crossentropy", learning_rate=0.000005, batch_size=1)
    model = tflearn.DNN(network, checkpoint_path="model_rt.tfl")
    model.load("model.tfl")
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=1, n_epoch=100) # currently set for retraining
    model.save("model_rt.tfl")
    
main()