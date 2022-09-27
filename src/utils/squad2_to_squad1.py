"""

This script coverts from a SQuAD v2.0 stanford format

{
"version": "v2.0",
"data": [
    {
    "title": str
    "paragraphs": [
        {
        "context": str,
        "qas" : [
            {
            "id": str,
            "question": str,
            "is_impossible": bool,
            "answers": [
                {
                "answer_start": int,
                "text": str
                },
            ]
            },
        ]
        },
    ]
    },
]
}

to a SQuAD v1.1 HuggingFace format

{
"version": "v1.1",
"data": [
    {
    "id": str,
    "title": str
    "context": str,
    "question": str,
    "answers":
        {
        "answer_start": [int, int, ...],
        "text": [str, str, ...]
        }
    },
]
}

In the process, impossible questions are filtered out

"""


import argparse
import json

from tqdm import tqdm


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('--with_impossible', action='store_true',)

    opt = parser.parse_args()

    with open(opt.input_path) as json_file:
        full_doc = json.load(json_file)

        data = full_doc['data']

    new_data = []

    for subject in tqdm(data):
        title = subject['title'] if 'title' in subject else ''
        paragraphs = subject['paragraphs']
        for paragraph in paragraphs:
            for qa in paragraph['qas']:
                if opt.with_impossible or 'is_impossible' not in qa or not qa['is_impossible']:

                    new_item = {
                        "context": paragraph["context"],
                        "title": title,
                        'question': qa['question'],
                        'id': qa['id']
                    }
                    if opt.with_impossible:
                        new_item['is_impossible'] = qa['is_impossible']

                    if opt.with_impossible and qa['is_impossible']:
                        new_item['plausible_answers'] = {'text': [a['text'] for a in qa['plausible_answers']],
                                                         'answer_start': [a['answer_start'] for a in qa['plausible_answers']]}
                    else:
                        new_item['answers'] = {'text': [a['text'] for a in qa['answers']], 'answer_start': [a['answer_start'] for a in qa['answers']]}
                    new_data.append(new_item)

    output_path = opt.input_path.replace('2.0', '1.1')
    with open(output_path, 'w') as json_out:
        full_doc['data'] = new_data
        full_doc['version'] = 'v1.1'
        json.dump(full_doc, json_out, ensure_ascii=False)
        print(f'file saved: {output_path}')