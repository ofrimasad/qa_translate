import argparse
import json
import os.path

from tqdm import tqdm


def squad2_to_squad2hf(_data: list) -> list:
    new_data = []

    for subject in tqdm(_data):
        title = subject['title'] if 'title' in subject else ''
        paragraphs = subject['paragraphs']
        for paragraph in paragraphs:
            for qa in paragraph['qas']:

                new_item = {
                    "context": paragraph["context"],
                    "title": title,
                    'question': qa['question'],
                    'id': qa['id']
                }

                if qa['is_impossible']:
                    new_item['answers'] = {'text': [], 'answer_start': []}
                else:
                    new_item['answers'] = {'text': [a['text'] for a in qa['answers']], 'answer_start': [a['answer_start'] for a in qa['answers']]}
                new_data.append(new_item)

    return new_data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)

    opt = parser.parse_args()

    if os.path.isdir(opt.input_path):
        input_paths = [os.path.join(opt.input_path, p) for p in os.listdir(opt.input_path) if 'v2.0hf' not in p]
    else:
        input_paths = [opt.input_path]


    for input_path in input_paths:
        with open(input_path) as json_file:
            full_doc = json.load(json_file)

            data = full_doc['data']

        new_data = squad2_to_squad2hf(data)

        output_path = input_path.replace('2.0', '2.0hf')
        with open(output_path, 'w') as json_out:
            full_doc['data'] = new_data
            full_doc['version'] = 'v2.0'
            json.dump(full_doc, json_out, ensure_ascii=False)
            print(f'file saved: {output_path}')