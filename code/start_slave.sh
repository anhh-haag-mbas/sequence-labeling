#!/usr/bin/env bash
set -x
export URL="http://172.31.91.37"
cd ~/sequence-labelling/code/
source .env/bin/activate
python3 master.py
