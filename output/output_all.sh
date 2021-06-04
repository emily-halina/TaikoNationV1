#!/bin/bash

for f in *.mp3; do python3 output.py "$f"; done

for f in *.osz; do mv "$f" "$PWD/output"; done