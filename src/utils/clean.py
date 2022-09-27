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



if __name__ == "__main__":

    a = '/home/ofri/qa_translate/data/squad/v2_translated/train-v2.0_de.json'

    with open(a) as json_file:
        full_doc = json.load(json_file)
        data_a = full_doc['data']

    for subject in data_a:

        paragraphs = subject['paragraphs']
        for paragraph in paragraphs:

            context = paragraph['context']

            qas = paragraph['qas']
            for qa in qas:
                question = qa['question']
                q_id = qa['id']

                if 'plausible_answers' in qa:
                    answers = qa['plausible_answers']
                else:
                    answers = qa['answers']

                for ans in answers:
                    ans_text = ans['text']
                    ans_text_words = ans_text.split(' ')




