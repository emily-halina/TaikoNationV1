'''
This program takes a .osu file and converts it into the "output" form
for our neural net. The output will be saved into a text file.

Hit Object Legend:
    - no note = 0
	- small red = 1
	- small blue = 2
	- big red = 3
	- big blue = 4
	- slider = 5
	- spinner = 6

Author: Emily Halina
'''

import sys
import numpy as np
from collections import deque

def main():
    if len(sys.argv) < 2:
        print("sorry i need a file here")
        return
    # open the file and dump the contents into a deque()
    f = open(sys.argv[1], "r", encoding="utf8")
    content = deque(f.read().splitlines())
    f.close()

    # extract the needed data for conversion
    filename, slider_multi, author_info = getName(content)
    timing = getTime(content)
    notes = getNotes(content, timing, slider_multi)

    # convert and write to file
    print(filename, "Conversion Success:", convert(notes, filename, author_info))
    return


def getName(content):
    '''
    Strips the first 4 sections from the .osu file, grabbing the filename & slider multiplier along the way.
    input: deque()
    output: str of the filename
    '''
    # strip the first two sections
    while content[0] != "[Metadata]":
        content.popleft()
    # get the file name (song_artist_difficulty)
    filename = "_".join([content[10][13:], content[1][6:], content[3][7:], content[6][8:]])
    filename = filename.replace(" ", '')
    filename = filename.replace("/", "_")
    author_info = "The creator of this beatmap is " + content[5][8:] + ". Check out the original beatmap at osu.ppy.sh/s/" + content[10][13:] + "\n"

    while content[0] != "[Difficulty]":
        content.popleft()
    slider_multi = float(content[5][17:])

    # clear out the metadata section to prep for getTime()
    while content[0] != "[TimingPoints]":
        content.popleft()
    content.popleft()
    return filename, slider_multi, author_info
    
def getTime(content):
    '''
    read through each timing point (comma seperated list) and store away each time & SV change for use in sliders / spinners
    input: deque()
    output: list of 3-tuples, in the format (time, beatLength, SVmultiplier)
    '''
    output = []
    while content[0] != "":
        line = content.popleft().split(',')
        time = int(float(line[0]))
        value = float(line[1])
        # uninhereted timing point
        if value >= 0:
            output.append( (time, value, 1) )
        else:
            output.append( (time, None, value) )
    
    return output

def getNotes(content, timing, slider_multi):
    '''
    read each line, determine what note it is, and save it in a tuple to be returned
    input: deque(), list of timings, slider_multi float for determining slider length
    output: list of tuples, organized as (time, hitObject, endPoint)

    time = int in time ms when note occurs
    hitObject = int from 0-10 inclusive determining what type of hitObject the note is, legend at top of script
    endPoint = int of time in ms when slider / spinner ends, None for dons/kats
    '''
    # get to the hitobjects
    while content[0] != "[HitObjects]":
        content.popleft()
    content.popleft()

    beatLength = timing[0][1]
    SV = timing[0][2]
    output = []

    # go through the rest of the file
    while len(content):
        time = None
        hitObject = None
        endPoint = None
        line = content.popleft().replace(":", ",").split(",")
        time = int(float(line[2]))
        objectType = int(line[3])
        # spinner
        if objectType == 12:
            hitObject = 8
            endPoint = int(float(line[5]))
        else:
            # check if the object is a slider
            try:
                sliderCheck = int(line[5])

            except:
                # slider, check out the time to make sure we're up to date on SV and beatLength
                i = 0
                while time <= timing[i][0]:
                    if timing[i][1] != None:
                        beatLength = timing[i][1]
                    SV = timing[i][2]
                    i += 1
                # calculate length with formula lengthvalue / (SliderMultiplier * 100) * beatLength
                hitObject = 5
                endPoint = float(line[len(line)-1]) * beatLength
                endPoint = endPoint / (slider_multi * SV * 100)
                endPoint = int(time + endPoint)

            else:
                # circle, check hitsound to see which kind
                hitsound = int(line[4])
                if hitsound == 0: # small red
                    hitObject = 1
                elif hitsound == 4: # big red
                    hitObject = 3
                elif hitsound == 2 or hitsound == 8 or hitsound == 10: # small blue
                    hitObject = 2
                elif hitsound == 6 or hitsound == 12 or hitsound == 14: # big blue
                    hitObject = 4
                else:
                    raise "HEY THIS CIRCLE DIDN'T GET READ!!!! WHY!!!!"

        output.append( (time, hitObject, endPoint) )
    return deque(output)

def convert(notes, filename, author_info):
    '''
    Takes the list of hitObjects in the song and transfers them to our desired format.
    input: deque() of 3-tuples containing note info (time, hitObject, endPoint), name of output file
    output: bool indicating success/failure
    '''
    # constants to determine what gets written to file, can be changed later
    NO_NOTE = "0\n"
    S_RED = "1\n"
    S_BLUE = "2\n"
    B_RED = "3\n"
    B_BLUE = "4\n"
    S_SLIDE = "5\n"
    E_SLIDE = "5\n"
    M_SLIDE = "5\n"
    S_SPIN = "6\n"
    E_SPIN = "6\n"
    M_SPIN = "6\n"
    INCREMENT = 23

    # for output to npy array
    zero = [1,0,0,0,0,0,0]
    one = [0,1,0,0,0,0,0]
    two = [0,0,1,0,0,0,0]
    three = [0,0,0,1,0,0,0]
    four = [0,0,0,0,1,0,0]
    five = [0,0,0,0,0,1,0]
    six = [0,0,0,0,0,0,1]
    output = []

    counts = {"no note":0,"s red":0,"s blue":0, "b red":0, "b blue":0, "slide":0, "spin":0}
    np_outfile = filename + ".npy"
    
    i = 0
    last_note = notes[len(notes)-1][0]
    success = True
    try:
        f = open(filename + '.txt', "w+")
        # write whatever header we use, currently just some author info / credits
        f.write(author_info)
        f.write("start\n")
        # move in increments of 23 ms
        # "time: " + str(i) + " " +
        # copy paste the above and add to print statements for timestamps (debugging)
        while len(notes):
            curr_note = notes.popleft()
            
            # move through dead space
            while (curr_note[0] - i) > INCREMENT:
                f.write(NO_NOTE)
                counts["no note"] += 1
                output.append(zero)
                i += INCREMENT
                
            # check the current note's object type
            hitObject = curr_note[1]
            # dons & kats
            if hitObject == 1:
                f.write(S_RED)
                counts["s red"] += 1
                output.append(one)
                i += INCREMENT
            elif hitObject == 2:
                f.write(S_BLUE)
                counts["s blue"] += 1
                output.append(two)
                i += INCREMENT
            elif hitObject == 3:
                f.write(B_RED)
                counts["b red"] += 1
                output.append(three)
                i += INCREMENT
            elif hitObject == 4:
                f.write(B_BLUE)
                counts["b blue"] += 1
                output.append(four)
                i += INCREMENT

            elif hitObject == 5:
                # slider has begun, work through the whole slider
                endPoint = curr_note[2]
                f.write(S_SLIDE)
                counts["slide"] += 1
                output.append(five)
                i += INCREMENT
                # fill in the middle of the slider
                while endPoint - i > INCREMENT:
                    f.write(M_SLIDE)
                    output.append(five)
                    counts["slide"] += 1
                    i += INCREMENT
                # slider has ended, cap it off
                f.write(E_SLIDE)
                counts["slide"] += 1
                output.append(five)
                i += INCREMENT
            else:
                # spinner has begun, work through the whole spinner
                endPoint = curr_note[2]
                f.write(S_SPIN)
                counts["spin"] += 1
                output.append(six)
                i += INCREMENT
                # fill in the middle of the spinner
                while endPoint - i > INCREMENT:
                    f.write(M_SPIN)
                    counts["spin"] += 1
                    output.append(six)
                    i += INCREMENT
                # spinner has ended, cap it off
                f.write(E_SPIN)
                counts["spin"] += 1
                output.append(six)
                i += INCREMENT
        f.write("end")
    except Exception as e:
        print(e.args[0])
        success = False
    finally:
        f.close()

    out_array = np.array(output, dtype=np.int32)
    print(len(out_array), "length of output array")
    np.save(np_outfile, out_array)
    print(counts)
    return success

if __name__ == "__main__":
    main()