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

import re
from tqdm import tqdm

from src.utils.smart_match import ModelMatcher

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)

    opt = parser.parse_args()
    matcher = ModelMatcher(model_name_or_path='/home/ofri/qa_translate/exp_xquad/train_ss_de') # 'onlplab/alephbert-base'
    match_thresh = 0.5

    with open(opt.input_path) as json_file:
        full_doc = json.load(json_file)

        data = full_doc['data']

    new_data = []
    multiple = 0
    for si, subject in enumerate(data):
        print(f'section {si}')
        title = subject['title'] if 'title' in subject else ''
        paragraphs = subject['paragraphs']
        for paragraph in tqdm(paragraphs):
            for qa in paragraph['qas']:
                if paragraph["context"] == "":
                    continue

                new_item = {
                    "context": paragraph["context"],
                    "title": title,
                    'question': qa['question'],
                    'id': qa['id']
                }

                if qa['is_impossible']:
                    new_item['answers'] = {'text': [],'answer_start': []}
                else:
                    new_item['answers'] = {'text': [],'answer_start': []}
                    for ans in qa['answers']:
                        if 'need_replace' in ans and ans['need_replace']:
                            if ans['text'] == "":
                                continue
                            translated_context = ans['translated_context']
                            translated_text = ans['translated_text']
                            translated_offset = ans['translated_offset']
                            new_answer, score = matcher.match(translated_context, translated_text)
                            if score > match_thresh:
                                translated_start_indices = [_.start() for _ in re.finditer(re.escape(new_answer), translated_context)]
                                if len(translated_start_indices) == 1:
                                    new_item['answers']['text'].append(new_answer)
                                    new_item['answers']['answer_start'].append(translated_start_indices[0] + translated_offset)
                                else:
                                    multiple+=1
                        else:
                            new_item['answers']['text'].append(ans['text'])
                            new_item['answers']['answer_start'].append(ans['answer_start'])

                if len(new_item['answers']['text']) > 0:
                    new_data.append(new_item)
    print(f'multiple: {multiple}')
    output_path = opt.input_path.replace('2.0', '2.0hf_05')
    with open(output_path, 'w') as json_out:
        full_doc['data'] = new_data
        full_doc['version'] = 'v2.0'
        json.dump(full_doc, json_out, ensure_ascii=False)
        print(f'file saved: {output_path}')