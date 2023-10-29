import json

def load_data(data_file):
    with open(data_file, "r") as data_file:
        data = json.load(data_file)
    return data

