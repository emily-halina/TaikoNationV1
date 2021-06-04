import numpy as np
import sys

def get_info(lines):

    ind = 0
    while lines[ind][:4] != "#OFF":
        ind += 1

    offset = float(lines[ind][8:len(lines[ind]) - 1]) * 100

    while lines[ind][:4] != "#BPM":
        ind += 1

    if lines[ind][len(lines[ind])-1] == ";":
        print("one bpm")
        i = 0
        while lines[ind][i] != "=":
            i += 1

        bpm = float(lines[ind][i+1 : len(lines[ind]) - 1])
        print(bpm)
    else:
        print("multi bpm")
        bpm = {}
        i = 0
        while lines[ind][i] != "=":
            i += 1

        bpm[0] = float(lines[ind][i+1 : len(lines[ind]) - 1])
        ind += 1
        while lines[ind][0] != ";":
            j = 0
            while lines[ind][j] != "=":
                j += 1
            key = float(lines[ind][1:j])
            value = float(lines[ind][j+1:len(lines[ind])])
            bpm[key] = value
            ind += 1
    print(bpm)

    return offset, bpm


def main():
    f = open(sys.argv[1], "r")
    lines = f.read().splitlines()
    f.close()

    bin_filename = sys.argv[1] + "_bin.npy"
    time_filename = sys.argv[1] + "_time.npy"

    offset, bpm = get_info(lines)
    multi_bpm = False
    
    if type(bpm) == dict:
        print("dict")
        line_time = 4 / (bpm[0] / 60)
        multi_bpm = True
    else:
        line_time = 4 / (bpm / 60)

    current_time = offset
    
    c = 0
    while lines[c][:2] != "//":
        c += 1
    
    while lines[c][0] != "0":
        c += 1

    curr = c
    count = 0
    for i in range(c, len(lines)):
        if lines[i] == "," or lines[i] == ";":
            count += 1

    print(count)
    note_time = []
    note_count = 0
    beat = 0
    bpm_swaps = 0

    for i in range(count):
        print("measure", i)
        m_count = 0
        measure = []
        while lines[curr] != "," and lines[curr] != ";":
            m_count += 1
            measure.append(lines[curr])
            curr += 1
        time = (line_time / m_count)*1000
        beat_div = 4 / m_count
        print(time)
        for note in measure:
            if "1" in note or "2" in note:
                note_time.append(current_time)
                print("NOTE AT ", current_time)
                note_count += 1
            current_time += time
            if multi_bpm and beat in bpm.keys():
                print("swapping bpm")
                line_time = 4 / (bpm[beat] / 60)
                time = (line_time / m_count)*1000
                bpm_swaps += 1
            beat += beat_div
        curr += 1

    print("note count", note_count)
    print("bpm swaps", bpm_swaps)
    bin_output = []
    out_time = 0
    n = 0
    for note in note_time:
        while abs(note - out_time) > 23:
            bin_output.append(0)
            out_time += 23
        bin_output.append(1)
        n += 1
        out_time += 23

    print(bin_output)
    print(note_time)
    print(len(note_time))
    print(len(bin_output))
    bin_output = np.array(bin_output)
    np.save(bin_filename, bin_output)
    np.save(time_filename, note_time)
    return

main()