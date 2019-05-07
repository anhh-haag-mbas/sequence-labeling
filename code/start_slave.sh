#!/usr/bin/env bash
# set -x
# export URL="http://172.31.91.37"
# cd ~/sequence-labelling/code/
# source .env/bin/activate
# python3 master.py
cd ~/sequence-labelling/ && git reset --hard HEAD && git pull && cd code && source .env/bin/activate && export URL="http://3.208.170.244:5000" && python3 master.py
