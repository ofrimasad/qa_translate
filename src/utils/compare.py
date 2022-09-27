import argparse
import json

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def find_in_b(data_b, q_id_a):
    for subject in data_b:

        paragraphs = subject['paragraphs']
        for paragraph in paragraphs:

            context = paragraph['context']

            qas = paragraph['qas']
            for qa in qas:
                question = qa['question']
                q_id = qa['id']
                if q_id == q_id_a:
                    answers = qa['answers']
                    return context, question, answers
    return "","",""



if __name__ == "__main__":

    b = '/home/ofri/qa_translate/data/squad/train-v2.0_de_aug2.json'
    a = '/home/ofri/qa_translate/data/squad/train-v2.0_de_aug3.json'
    en = '/home/ofri/qa_translate/data/squad/train-v2.0.json'

    diff_a = '/home/ofri/qa_translate/data/diffs/train-v2.0_de_aug2_3.txt'
    diff_b = '/home/ofri/qa_translate/data/diffs/train-v2.0_de_aug3_2.txt'

    with open(a) as json_file:
        full_doc = json.load(json_file)
        data_a = full_doc['data']

    with open(b) as json_file:
        full_doc = json.load(json_file)
        data_b = full_doc['data']

    with open(en) as json_file:
        full_doc = json.load(json_file)
        data_en = full_doc['data']


    with open(diff_a, 'w') as diff_a_file:
        with open(diff_b, 'w') as diff_b_file:
            for subject in data_a:

                paragraphs = subject['paragraphs']
                for i, paragraph in enumerate(paragraphs):

                    context = paragraph['context']
                    if context is None:
                        continue

                    context_words = context.split(' ')

                    qas = paragraph['qas']
                    for qa in qas:
                        question = qa['question']
                        q_id = qa['id']

                        context_b, q_b, as_b = find_in_b(data_b, q_id)
                        context_en, q_en, as_en = find_in_b(data_en, q_id)

                        if context != context_b and context_b != "":
                            print(f'{i}\n CONTEXT A:\n{context}\nCONTEXT B:\n{context_b}\nCONTEXT EN:\n{context_en}\nQUESTION A:\n{question}\nQUESTION B:\n{q_b}')
                            diff_a_file.write(f'{context}\n')
                            diff_b_file.write(f'{context_b}\n')

                        if 'plausible_answers' in qa:
                            answers = qa['plausible_answers']
                        else:
                            answers = qa['answers']

                        for ans in answers:
                            ans_text = ans['text']
                            ans_text_words = ans_text.split(' ')

                        # for ans_a, ans_b in zip(answers, as_b):
                        #     print(f'ANS A:\n{ans_a["text"]} {ans_a["answer_start"]}\nANS B:\n{ans_b["text"]} {ans_b["answer_start"]}\n')


