import argparse
import json
import os

from tqdm import tqdm


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, required=True)
    parser.add_argument('-o', '--output_path', type=str, required=True)

    opt = parser.parse_args()

    with open(opt.input_path) as json_file:
        full_doc = json.load(json_file)
        data = full_doc['data']

    new_data = [{'paragraphs': data}]

    for subject in tqdm(new_data):
        subject['title'] = 'title'
        paragraphs = subject['paragraphs']
        for paragraph in paragraphs:

            answers = [{'text': t, 'answer_start': s} for t, s in zip(paragraph['answers']['text'], paragraph['answers']['answer_start'])]
            paragraph['qas'] = [{'question': paragraph['question'], 'answers': answers, 'id': paragraph['id']}]
            paragraph.pop('question')
            paragraph.pop('answers')
            paragraph.pop('id')

    with open(opt.output_path, 'w') as json_out:
        full_doc['data'] = new_data
        json.dump(full_doc, json_out, ensure_ascii=False)
        print(f'file saved: {opt.output_path}')