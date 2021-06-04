from pydub import AudioSegment
import os
import sys



def process_song(song_name):
    no_prefix = song_name[:len(song_name) - 4]
    print(no_prefix)
    song = AudioSegment.from_mp3(song_name)
    seg_length = 23
    current_segment = song[:seg_length]
    directory_name = song_name + " chunks"
    print(len(current_segment))

    try:
        os.mkdir(directory_name)
    except:
        print("directory exists, skipping this one")
        return
    else:
        print("directory", directory_name, "created")

    i = 1
    current_segment.export(directory_name + "/" + no_prefix + ' ' + str(i) + ".wav", format = "wav", bitrate = "192k")
    i += 1
    
    while len(current_segment) == 23:
        current_segment = song[seg_length*i : seg_length*i + seg_length]
        current_segment.export(directory_name + "/" + no_prefix + ' ' + str(i) + ".wav", format = "wav", bitrate = "192k")
        i += 1

    # cap it off
    current_segment.export(directory_name + "/" + no_prefix + ' ' + str(i) + ".wav", format = "wav", bitrate = "192k")
    print("success! number of segments:", str(i))


