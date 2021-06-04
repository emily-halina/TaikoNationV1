#!/bin/bash

for f in *.osu; do python3 convert_new_rep.py "$f"; done