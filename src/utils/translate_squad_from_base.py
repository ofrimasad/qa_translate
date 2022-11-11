import argparse
import json

import re

import os
from tqdm import tqdm

from utils.smart_match import ModelMatcher, CorrelationMatcher


def hash_original(original_dataset_path):
    with open(original_dataset_path) as original_file:
        original_doc = json.load(original_file)

    original_data = original_doc['data']
    orig_dict = {}

    for si, subject in enumerate(original_data):
        paragraphs = subject['paragraphs']
        for paragraph in paragraphs:
            for qa in paragraph['qas']:
                assert qa['id'] not in orig_dict
                orig_dict[qa['id']] = qa

    return orig_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('base_input_path', type=str, help='path of the base file')
    parser.add_argument('lang', type=str, help='language symbol')
    parser.add_argument('model_path', type=str, help='path of the trained matcher')
    parser.add_argument('--original_dataset_path', type=str, default='/home/ofri/qa_translate/data/squad/train-v2.0.json')
    parser.add_argument('--match_thresh', type=float, default=0.95, help='threshold for matcher')
    parser.add_argument('--output_dir', type=str, help='path for output dir')
    parser.add_argument('--from_en', action='store_true', help='match from english')
    parser.add_argument('--correlation', action='store_true', help='use correlation matcher')

    opt = parser.parse_args()

    for k, v in opt.__dict__.items(): print(f'{k}:\t{v}')

    if opt.correlation:
        matcher = CorrelationMatcher('bert-base-multilingual-cased')
    else:
        matcher = ModelMatcher(model_name_or_path=opt.model_path)

    with open(opt.base_input_path) as json_file:
        full_doc = json.load(json_file)

    if opt.output_dir:
        os.makedirs(opt.output_dir, exist_ok=True)

    orig_dict = hash_original(opt.original_dataset_path)

    data = full_doc['data']
    new_data = []
    multiple = 0
    require_translation, not_require_translation, success = 0,0,0

    from_en = opt.from_en
    for si, subject in enumerate(tqdm(data)):
        title = subject['title'] if 'title' in subject else ''
        paragraphs = subject['paragraphs']

        for paragraph in paragraphs:

            for qa in paragraph['qas']:

                # drop empty context
                if paragraph["context"] == "":
                    continue

                new_item = {
                    "context": paragraph["context"],
                    "title": title,
                    'question': qa['question'],
                    'id': qa['id']
                }

                if qa['is_impossible']:
                    new_item['answers'] = {'text': [], 'answer_start': []}
                else:
                    new_item['answers'] = {'text': [], 'answer_start': []}
                    for ans in qa['answers']:
                        if 'need_replace' in ans and ans['need_replace']:
                            require_translation += 1
                            # drop empty answer
                            if ans['text'] == "":
                                continue

                            translated_context = ans['translated_context']
                            original_text = ans['original_text'] if from_en else ans['translated_text'] #orig_dict[qa['id']]['answers'][0]['text']
                            translated_offset = ans['translated_offset']
                            new_answer, score = matcher.match(translated_context, original_text)

                            if score > opt.match_thresh:
                                success += 1
                                translated_start_indices = [_.start() for _ in re.finditer(re.escape(new_answer), translated_context)]
                                if len(translated_start_indices) == 1:
                                    new_item['answers']['text'].append(new_answer)
                                    new_item['answers']['answer_start'].append(translated_start_indices[0] + translated_offset)
                                else:
                                    multiple += 1
                            else:
                                multiple += 1
                        else:
                            not_require_translation += 1
                            new_item['answers']['text'].append(ans['text'])
                            new_item['answers']['answer_start'].append(ans['answer_start'])

                if len(new_item['answers']['text']) > 0:
                    new_data.append(new_item)

    print(f'multiple: {multiple}')

    phase = 'dev' if 'dev' in opt.base_input_path else 'train'
    out_dir = opt.output_dir or os.path.dirname(opt.base_input_path)
    output_path = f'{out_dir}/{phase}_v1.0hf_{opt.lang}_{opt.match_thresh:.2f}{"_enq" if opt.from_en else ""}.json'
    print(f'require: {require_translation}')
    print(f'not require {not_require_translation}')
    print(f'success: {success}')
    with open(output_path, 'w') as json_out:
        full_doc['data'] = new_data
        full_doc['version'] = 'v1.0'
        json.dump(full_doc, json_out, ensure_ascii=False)
        print(f'file saved: {output_path}')
