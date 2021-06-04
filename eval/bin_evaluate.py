'''
evaluation script for binary representation

checks given files vs random noise, as well as vs human benchmark

press 1 for full folder, press 2 for just one (enter them as program args)
'''

import numpy as np
import sys
import os

def main():
    # initialize random noise, can put seed here if needed!
    rng = np.random.default_rng(2009000042)
    noise = rng.integers(low=0, high=2, size=1000000)
    mode = input("entire folder or single given input? (1/2)")
    if mode == "1":
        # set up our stat trackers
        file_list = os.listdir()
        index = 0
        noise_sum = 0
        human_sum = 0
        human_sum2 = 0
        c_v_c = 0
        ai_slide_sum = 0
        hum_slide_sum = 0

        # go through the file list, comparing each ai chart to its human counterpart
        for i in range(len(file_list)):
            if file_list[index][len(file_list[index]) - 3:] == "npy":
                ai_file = file_list[index]
                human_file = file_list[index + 1]
                ai_chart = np.load(ai_file)
                human_chart = np.load(human_file)
                
                print(ai_file, "versus random noise:")
                noise_sum += vsRandom(ai_chart, noise)
            
                print(ai_file, "versus", human_file)
                human_sum += vsHuman(noise, human_chart)
                human_sum2 += vsHuman3(noise, human_chart)
                print(ai_file, "sliding scale")
                ai_slide_sum += slidingScale(ai_chart)

                print(human_file, "sliding scale")
                hum_slide_sum += slidingScale(human_chart)

                print("chart vs chart")
                c_v_c += slidingScaleV(ai_chart, human_chart)

            else:
                break
            
            index += 2

        noise_sum /= 10
        human_sum /= 10
        ai_slide_sum /= 10
        hum_slide_sum /= 10
        c_v_c /= 10
        human_sum2 /= 10

        print("RESULTS:")
        print(noise_sum, "percent average similarity to noise (DC RANDOM)")
        print(human_sum, "percent average similarity to human (DC HUMAN)")
        print(human_sum2, "percent avg similarity to human with slide scale (OC HUMAN)")
        print(ai_slide_sum, "ai pattern score", hum_slide_sum, "human pattern score (OVER P-SPACE)")
        print(c_v_c, "percentage of human patterns ai used (HI P-SPACE")

    elif mode == "2":
        ai_file = sys.argv[1]
        human_file = sys.argv[2]
        ai_chart = np.load(ai_file)
        human_chart = np.load(human_file)

        print(ai_file, "versus Random Noise:")
        vsRandom(ai_chart, noise)

        print(ai_file, "versus", human_file)
        vsHuman(ai_chart, human_chart)
    return

def vsRandom(chart, noise):
    # compares a given chart to random noise by similarities vs total
    total = len(chart)
    similarity = 0
    for i in range(total):
        if chart[i] == noise[i]:
            similarity += 1
    result = (similarity / total)*100
    print(result, "percent similar\n")
    return result

def vsHuman(chart, chart2):
    # compares two charts by similarities vs total
    limit = min(len(chart), len(chart2))
    start = 0
    similarity = 0
    over = 0
    under = 0 

    while chart2[start] != 1:
        start += 1

    total = limit - start
    for i in range(start, limit):
        if chart[i] == chart2[i]:
            similarity += 1
        elif chart[i] == 0:
            under += 1
        else:
            over += 1
    
    result = (similarity / total)*100
    under = (under / total)*100
    over = (over / total)*100
    print(result, "percent similar", under, "percent underchart", over, "percent overchart\n")
    return result


def slidingScale(chart):
    # returns the percent of the possibility space that chart is covering over scale
    # SCALE * 23 = # of ms, so 4 = 92ms, 8 = 184, 16 = 368, ..
    SCALE = 8
    
    patterns = set()
    last_ind = len(chart) - SCALE + 1
    for i in range(last_ind):
        chunk = tuple(chart[i:i+SCALE])
        if chunk not in patterns:
            patterns.add(chunk)
        
    
    p_score = (len(patterns) / 2**(SCALE))*100 # scale by space of possibilities
    print(p_score, "pattern score\n")
    return p_score


def slidingScaleV(chart, chart2):
    # returns the percent of the possibility space that chart is covering over scale
    # SCALE * 23 = # of ms, so 4 = 92ms, 8 = 184, 16 = 368, ..
    SCALE = 8
    
    patterns1 = set()
    patterns2 = set()
    last_ind1 = len(chart) - SCALE + 1
    last_ind2 = len(chart2) - SCALE + 1

    for i in range(last_ind1):
        chunk = tuple(chart[i:i+SCALE])
        if chunk not in patterns1:
            patterns1.add(chunk)
    
    for i in range(last_ind2):
        chunk = tuple(chart2[i:i+SCALE])
        if chunk not in patterns2:
            patterns2.add(chunk)
    
    p_score = len(patterns1.intersection(patterns2))  
    p_score = (p_score / len(patterns2))*100
    print(p_score, "pattern score\n")
    return p_score


def vsHuman3(chart, chart2):
    # compares two charts by similarities with a "buffer" for hit notes
    limit = min(len(chart), len(chart2))
    start = 0
    similarity = 0
    buffer = 1


    while chart2[start] != 1:
        start += 1

    total = 0
    total = limit - start
    for i in range(start, limit):
        if chart[i] == 1:
            b = -buffer
            while b <= buffer:
                try:
                    if chart2[i + b] == 1:
                        similarity += 1
                        break
                except Exception:
                    print("uwu")
                b += 1
        elif chart[i] == 0:
            if chart[i] == chart2[i]:
                similarity += 1
    
    result = (similarity / total)*100
    print(result, "percent similar\n")
    return result

if __name__ == "__main__":
    main()