import tflearn
import tensorflow as tf
import numpy as np
import sys
import os
import random
from essentia.standard import MonoLoader, Windowing, Spectrum, MelBands
from collections import deque
from zipfile import ZipFile
import csv

from chop_song import process_song


def main():
    '''
    process the given song, analyze it, then create a chart for it
    Usage: output.py song_file.mp3
    '''
    if len(sys.argv) < 2:
        print("Usage: output.py song_file.mp3")
        return

    dir_name = sys.argv[1] + " chunks"

    songfile = sys.argv[1]
    song = sys.argv[1][:len(sys.argv[1]) - 4]
    npy_file = song + " Input.npy"
    outfile = song + " TaikoNation.osu"
    package = song + " TaikoNation.osz"
    dir_list = os.listdir()

    if dir_name not in dir_list:
        print("Chopping up your song..")
        process_song(sys.argv[1])
    else:
        print("Song already chopped, moving on..")

    print("Analyzing your song..")
    analyze_song(npy_file, dir_name)

    print("Making predictions..")
    make_predictions(npy_file, outfile)

    print("Packing up your beatmap..")
    create_osz(songfile, outfile, package)
    print("All done!", package, "created in current directory.")
    return

def create_analyzers(fs=44100.0,
                     nhop=1024,
                     nffts=[1024, 2048, 4096],
                     mel_nband=80,
                     mel_freqlo=27.5,
                     mel_freqhi=16000.0):
    '''
    create analyzer from DDC, adapted to TaikoNation
    https://arxiv.org/abs/1703.06891
    '''
    analyzers = []
    for nfft in nffts:
        window = Windowing(size=nfft, type='blackmanharris62')
        spectrum = Spectrum(size=nfft)
        mel = MelBands(inputSize=(nfft // 2) + 1,
                       numberBands=mel_nband,
                       lowFrequencyBound=mel_freqlo,
                       highFrequencyBound=mel_freqhi,
                       sampleRate=fs)
        analyzers.append((window, spectrum, mel))
    return analyzers[0][0], analyzers[0][1], analyzers[0][2]

def analyze_song(file_name = None, dir_name = None):
        '''
        write something here
        '''
        file_list = os.listdir()
        for f in file_list:
            if f == file_name:
                print("Song already has been processed, exiting processing..")
                #os.chdir(cwd)
                return

        cwd = os.getcwd()
        if dir_name != None:
            new_dir = cwd + "/" + dir_name
            os.chdir(new_dir)

        file_list = os.listdir()
        window, spectrum, mel = create_analyzers()
        feats_list = []
        i = 0
        
        for fn in file_list:
            if fn[len(fn) - 1] != 'v':
                continue
            try:
                loader = MonoLoader(filename=fn, sampleRate=44100.0)
                samples = loader()
                feats = window(samples)
                if len(feats) % 2 != 0:
                    feats = np.delete(feats, random.randint(0, len(feats) - 1))
                feats = spectrum(feats)
                feats = mel(feats)
                feats_list.append(feats)
                i+=1
            except Exception as e:
                feats_list.append(np.zeros(80, dtype=np.float32))
                i += 1

        # Apply numerically-stable log-scaling
        feats_list = np.array(feats_list)
        feats_list = np.log(feats_list + 1e-16)
        print(len(feats_list), "length of feats list")
        print(type(feats_list[0][0]))
        if dir_name != None:
            os.chdir(cwd)
        np.save(file_name, feats_list)
        return
        


def make_predictions(npy_file=None, outfile=None):
    '''
    Makes note predictions using the given song data (npy_file), then calls create_chart to... create the chart!
    '''

    ### importing model for predictions ###
    
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

    song_encoder = tflearn.conv_1d(song, nb_filter=32, filter_size=3, activation="relu")
    song_encoder = tflearn.max_pool_1d(song_encoder, kernel_size=2)

    song_encoder = tflearn.fully_connected(song_encoder, n_units=128, activation="relu")
    song_encoder = tf.reshape(song_encoder, [-1,8,16])

    # split song data into past chunks and current chunk
    past_chunks = tf.slice(song_encoder, [0,0,0], [-1, 8, 15])
    curr_chunk = tf.slice(song_encoder, [0,0,15], [-1, 8, 1])

    # combine note data with processed song data
    lstm_input = tf.unstack(past_chunks, axis=1)
    lstm_input = tf.math.multiply(lstm_input, prev_notes)
    lstm_input = tf.reshape(lstm_input, [-1]) # flatten this to add on the current chunk
    
    # add on the final segment which does not have data yet
    curr_chunk = tf.math.multiply(curr_chunk, tf.ones([8, 15]))
    curr_chunk = tf.reshape(curr_chunk, [-1])
    lstm_input = tf.concat([lstm_input, curr_chunk], 0)

    lstm_input = tf.reshape(lstm_input, [-1, 16, 88]) # reshape to desired shape
    
    # 2 lstm layers, then a final fully connected softmax layer
    lstm_input = tflearn.lstm(lstm_input, 64, dropout=0.8, activation="relu")
    lstm_input = tf.reshape(lstm_input, [-1, 8, 8])

    lstm_input = tflearn.lstm(song_encoder, 64, dropout=0.8, activation="relu")

    lstm_input = tflearn.fully_connected(lstm_input, n_units=28, activation="softmax")
    lstm_input = tflearn.reshape(lstm_input, [-1,4,7])

    # setting up final parameters
    network = tflearn.regression(lstm_input, optimizer = "adam", loss="categorical_crossentropy", learning_rate=0.0001, batch_size=1)
    model = tflearn.DNN(network, checkpoint_path="nr_model")

    cwd = os.getcwd()
    model_dir = cwd + "/model"
    os.chdir(model_dir)
    model.load("model.tfl")
    os.chdir(cwd)

    # load the song data from memory map and reshape appropriately        
    song_mm = np.load(npy_file, mmap_mode="r")
    song_data = np.frombuffer(buffer=song_mm, dtype=np.float32, count=-1)
    song_data = song_data[0:song_mm.shape[0]*song_mm.shape[1]]
    song_data = np.reshape(song_data, song_mm.shape)
    
    # create the given song chunk
    # predict for the current song chunk
    # feed this prediction information back into the model

    note_queue = deque([])
    for i in range(16):
        note_queue.append(np.zeros(7))
    
    predictions = []

    while len(note_queue) != 16 and len(note_queue) < 16:
        note_queue.append(np.zeros(7))
    for j in range(len(song_data)):
        input_chunk = []
        song = []
        note_data = []

        for i in range(16):
            if j - i < 0:
                song.append(np.zeros(80))
            else:
                song.append(song_data[j-i])
            if i < 12:
                note_data.append(note_queue[i])
            elif i != 15:
                note_data.append(np.ones(7))

        song = np.array(song).flatten()
        note_data = np.array(note_data).flatten()
        input_chunk = np.concatenate([song, note_data])
        input_chunk = np.expand_dims(input_chunk, axis=0)
        p = model.predict(input_chunk)
        note_queue.popleft()
        predictions.append(p[0])
        note_queue.append(p[0][0])

    note_selections = []
    ## REMOVE ME
    f = open("predictions.csv", "a+", newline='')
    writer = csv.writer(f)
    for k in range(len(predictions)):
        guess = np.zeros([7])
        for n in range(4):
            try:
                selection = np.array(predictions[k+n][3-n])
                guess = np.add(guess, selection)
            except IndexError:
                # we have reached the end
                break
        prob = guess / np.sum(guess)
        writer.writerow(prob)
        try:
            choice = np.random.choice([0,1,2,3,4,5,6], p=prob)
            note_selections.append(choice)
        except ValueError:
            print('uh oh')
    
    create_chart(note_selections, outfile)
    return

def create_chart(note_selections, file_name="outfile.osu"):
    '''
    Create the .osu file based on the note selections
    '''
    # template for beginning of file
    osu_file = """osu file format v14

[General]
AudioFilename: audio.mp3
AudioLeadIn: 0
PreviewTime: 0
Countdown: 0
SampleSet: Normal
StackLeniency: 0.7
Mode: 1
LetterboxInBreaks: 0
WidescreenStoryboard: 0

[Editor]
DistanceSpacing: 0.8
BeatDivisor: 4
GridSize: 32
TimelineZoom: 3.14

[Metadata]
Title:SongTitle
TitleUnicode:SongTitle
Artist:ArtistName
ArtistUnicode:ArtistName
Creator:TaikoNation
Version:TaikoNation v1
Source:
Tags:
BeatmapID:-1
BeatmapSetID:-1

[Difficulty]
HPDrainRate:6
CircleSize:2
OverallDifficulty:6
ApproachRate:10
SliderMultiplier:1.4
SliderTickRate:1

[TimingPoints]
0,368,4,1,0,40,1,0


[HitObjects]
"""
    current_ms = 0
    last_note_active = False
    outfile = open(file_name, "w+")
    # add each note to the string with its corresponding time
    for note in note_selections:
        if note == 1 and last_note_active == False:
            osu_file += ("256,192," + str(current_ms) + ",1,0,0:0:0:0:\n")
            last_note_active = True
        elif note == 2 and last_note_active == False:
            osu_file += ("256,192," + str(current_ms) + ",1,2,0:0:0:0:\n")
            last_note_active = True
        elif note == 3 and last_note_active == False:
            osu_file += ("256,192," + str(current_ms) + ",1,4,0:0:0:0:\n")
            last_note_active = True
        elif note == 4 and last_note_active == False:
            osu_file += ("256,192," + str(current_ms) + ",1,6,0:0:0:0:\n")
            last_note_active = True
        else:
            last_note_active = False
        current_ms += 23
    outfile.write(osu_file)
    return

def create_osz(songfile, outfile, package):
    '''
    Package the song .mp3 (songfile) and .osu chart data (outfile) into a single .osz which can be dragged into osu for instant use (package)
    '''
    # set up
    temp_name = songfile
    os.rename(songfile, "audio.mp3")
    # zip up
    with ZipFile(package, mode="w") as oszf:
        oszf.write("audio.mp3")
        oszf.write(outfile)
    # clean up
    os.rename("audio.mp3", temp_name)
    os.remove(outfile)
    return
main()