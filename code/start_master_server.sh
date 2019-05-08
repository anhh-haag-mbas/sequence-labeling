#!/usr/bin/env bash
set -x
cd ~/sequence-labelling/code/
source .env/bin/activate
python3 master_server.py
