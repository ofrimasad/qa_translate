import argparse
import json

from tqdm import tqdm


if __name__ == "__main__":

    """
    remove the impossibles from a dataset
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)

    opt = parser.parse_args()

    with open(opt.input_path) as json_file:
        full_doc = json.load(json_file)
        data = full_doc['data']

    new_data = []
    for paragraph in tqdm(data):
        if len(paragraph['answers']['answer_start']) > 0:
            new_data.append(paragraph)

    with open(opt.output_path, 'w') as json_out:
        full_doc['data'] = new_data
        json.dump(full_doc, json_out, ensure_ascii=False)
        print(f'file saved: {opt.output_path}')