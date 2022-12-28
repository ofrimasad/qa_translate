import argparse
import json

import re

import os

import numpy as np
from matplotlib import pyplot as plt
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
                orig_dict[qa['id']] = {"qa": qa, "context": paragraph['context']}

    return orig_dict

def draw_hist(data: list, title, wc):
    counts, bins = np.histogram(data, bins=np.arange(40))

    counts_wc, bins_wc = np.histogram(wc, bins=np.arange(40))
    counts = np.clip(counts / (counts_wc + 0.0000001), 0, 1)

    plt.stairs(counts, bins, fill=True)
    plt.title(title)
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('base_input_path', type=str, help='path of the base file')
    parser.add_argument('lang', type=str, help='language symbol')
    parser.add_argument('model_path', type=str, help='path of the trained matcher')
    parser.add_argument('--original_dataset_path', type=str, default='/home/ofri/qa_translate/data/squad/train-v2.0.json')
    parser.add_argument('--match_thresh', type=float, default=0.95, help='threshold for matcher')
    parser.add_argument('--output_dir', type=str, help='path for output dir')
    parser.add_argument('--from_en', action='store_true', help='match from english')
    parser.add_argument('--augment', action='store_true', help='match from english')
    parser.add_argument('--correlation', action='store_true', help='use correlation matcher')
    parser.add_argument('--both', action='store_true', help='use both type of matchers')

    opt = parser.parse_args()

    # assert not (opt.augment and opt.from_en), "can't augment when using from english"


    for k, v in opt.__dict__.items(): print(f'{k}:\t{v}')

    if opt.correlation:
        matcher = CorrelationMatcher('bert-base-multilingual-cased')
    else:
        matcher = ModelMatcher(model_name_or_path=opt.model_path)

    if opt.both:
        cor_matcher = CorrelationMatcher('bert-base-multilingual-cased')

    with open(opt.base_input_path) as json_file:
        full_doc = json.load(json_file)

    if opt.output_dir:
        os.makedirs(opt.output_dir, exist_ok=True)

    orig_dict = hash_original(opt.original_dataset_path)

    data = full_doc['data']
    new_data = []
    multiple = 0
    cant_find = 0
    wc_success = []
    wc_fail = []
    wc_success_cor = []
    wc_fail_cor = []
    wc = []
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

                if 'is_impossible' in qa and qa['is_impossible']:
                    new_item['answers'] = {'text': [], 'answer_start': []}
                else:
                    new_item['answers'] = {'text': [], 'answer_start': []}
                    # assert len(qa['answers']) == 1
                    for ans in qa['answers']:
                        if 'need_replace' in ans and ans['need_replace']:
                            require_translation += 1
                            # drop empty answer
                            if ans['text'] == "":
                                continue

                            translated_context = ans['translated_context']
                            if translated_context is None or translated_context == "":
                                continue
                            original_text = ans['original_text'] if from_en else ans['translated_text'] #orig_dict[qa['id']]['answers'][0]['text']
                            translated_text = ans['translated_text']
                            translated_offset = ans['translated_offset']
                            new_answer, score = matcher.match(translated_context, original_text)
                            cor_score = 0
                            length = len(original_text.split(" "))
                            # wc.append(length)
                            # if score > opt.match_thresh:
                            #     wc_success.append(length)
                            # else:
                            #     wc_fail.append(length)

                            # new_answer, score = cor_matcher.match(translated_context, original_text)
                            # if score > 0.5:
                            #     wc_success_cor.append(l)
                            # else:
                            #     wc_fail_cor.append(l)

                            # if long sentence and failed to match
                            if opt.both and score < opt.match_thresh and length > 15:
                                new_answer, cor_score = cor_matcher.match(translated_context, translated_text)

                            if score > opt.match_thresh or cor_score > 0.5:

                                translated_start_indices = [_.start() for _ in re.finditer(re.escape(new_answer), translated_context)]
                                if len(translated_start_indices) == 1:
                                    success += 1
                                    if opt.augment:
                                        new_item['answers']['text'].append(translated_text)
                                        new_item['answers']['answer_start'].append(translated_start_indices[0] + translated_offset)
                                        cont = new_item['context']
                                        cont = cont[:new_item['answers']['answer_start'][-1]] + new_item['answers']['text'][-1] + cont[new_item['answers']['answer_start'][-1] + len(new_answer):]
                                        new_item['context'] = cont
                                    else:
                                        new_item['answers']['text'].append(new_answer)
                                        new_item['answers']['answer_start'].append(translated_start_indices[0] + translated_offset)
                                else:
                                    multiple += 1
                            else:
                                cant_find += 1
                        else:
                            not_require_translation += 1
                            new_item['answers']['text'].append(ans['text'])
                            new_item['answers']['answer_start'].append(ans['answer_start'])

                if len(new_item['answers']['text']) > 0:
                    new_data.append(new_item)

    print(f'multiple: {multiple}')
    print(f'cant_find: {cant_find}')
    phase = 'dev' if 'dev' in opt.base_input_path else 'train'
    out_dir = opt.output_dir or os.path.dirname(opt.base_input_path)
    output_path = f'{out_dir}/{phase}_v1.0hf_{opt.lang}_{opt.match_thresh:.2f}{"_enq" if opt.from_en else ""}{"_aug" if opt.augment else ""}{"_both" if opt.both else ""}.json'
    print(f'require: {require_translation}')
    print(f'not require {not_require_translation}')
    print(f'success: {success}')

    # wc = np.array(wc)
    # draw_hist(np.array(wc_success) , "wc_success", wc)
    # draw_hist(np.array(wc_fail), "wc_fail", wc)
    # draw_hist(np.array(wc_success_cor) , "wc_success_cor", wc)
    # draw_hist(np.array(wc_fail_cor) , "wc_fail_cor", wc)
    # squad_statistics_hf.draw_hist(wc, "wc")
    # extract_stats(new_data)
    with open(output_path, 'w') as json_out:
        full_doc['data'] = new_data
        full_doc['version'] = 'v1.0'
        json.dump(full_doc, json_out, ensure_ascii=False)
        print(f'file saved: {output_path}')
