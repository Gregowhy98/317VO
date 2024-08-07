
import json


def load_config(config_path):
    with open(config_path) as json_file:
        return json.load(json_file)    
    
def eval_loop():
    pass

def test_loop():
    pass