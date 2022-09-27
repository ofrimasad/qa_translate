import argparse
import json

import random

if __name__ == "__main__":
    """
    Extract a random sample with given size from a dataset
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, required=True)
    parser.add_argument('-o', '--output_path', type=str, required=True)
    parser.add_argument('-s', '--size', type=int, required=True)

    opt = parser.parse_args()

    with open(opt.input_path) as json_file:
        full_doc = json.load(json_file)
        data = full_doc['data']

    new_data = random.sample(data, opt.size)

    with open(opt.output_path, 'w') as json_out:
        full_doc['data'] = new_data
        json.dump(full_doc, json_out, ensure_ascii=False)
        print(f'file saved: {opt.output_path}')