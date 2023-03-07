import argparse
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_a', type=str)
    parser.add_argument('input_b', type=str)
    parser.add_argument('output', type=str)

    opt = parser.parse_args()

    with open(opt.input_a) as json_file:
        full_doc = json.load(json_file)
        data_a = full_doc['data']

    with open(opt.input_b) as json_file:
        full_doc = json.load(json_file)
        data_b = full_doc['data']

    full_doc['data'] = data_a + data_b
    with open(opt.output, "w") as json_file:
        json.dump(full_doc, json_file)

