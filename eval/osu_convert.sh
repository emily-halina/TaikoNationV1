#!/bin/bash

for f in *.osu; do python3 osu_to_bin.py "$f"; done

for f in *bin.npy; do mv "$f" "$PWD/converted_charts/ai_taiko_bin"; done

for f in *.npy; do mv "$f" "$PWD/converted_charts/ai_taiko_time"; done