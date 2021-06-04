#!/bin/bash

for f in *.sm; do python3 sm_to_bin.py "$f"; done

for f in *bin.npy; do mv "$f" "$PWD/converted_charts/ai_sm_bin"; done

for f in *.npy; do mv "$f" "$PWD/converted_charts/ai_sm_time"; done