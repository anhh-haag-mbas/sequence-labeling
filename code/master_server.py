from flask import Flask
from flask import request
from threading import Lock
from random import choice
import json

def read_configs():
    configs = []
    with open('master_server_configs', 'r') as f:
        for line in f:
            configs += [eval(line)]
    return configs

configs = read_configs()
configs_lock = Lock()

def get_config():
    global configs
    configs_lock.acquire()
    if len(configs) > 0:
        config = choice(configs)
        configs.remove(config)
    else:
        config = None
    configs_lock.release()
    return config

def append_sent_config(config):
    with open('master_server_sent', 'a') as f:
        f.write(repr(config) + "\n")

def save_result(result):
    with open('master_server_results', 'a') as f:
        f.write(result)

app = Flask(__name__)

@app.route("/")
def get_conf():
    return "please GET /configuration or POST /result"

@app.route("/configuration")
def get_configuration():
    config = get_config()
    append_sent_config(config)
    return json.dumps(config)

@app.route("/result", methods=['POST'])
def post_result():
    json = request.get_json()
    if json is None:
        return 'No JSON got', 500
    else:
        save_result(json)
        return "Saved: " + json
