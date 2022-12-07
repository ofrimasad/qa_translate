import argparse
import json

import os

import fnmatch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

questions_first_words = {'what': 0, 'which': 0, 'who': 0, 'when': 0, 'where': 0, 'how much': 0, 'how many': 0, 'how': 0, 'why': 0, 'other': 0}


def count_q_word(question: str):
    question = question.lower()
    found = False
    for k in questions_first_words.keys():
        if k in question:
            questions_first_words[k] += 1
            found = True
            break
    if not found:
        questions_first_words['other'] += 1


def print_dict(dictionary: dict, percentage: bool = False):
    if percentage:
        total = 0
        for v in dictionary.values():
            total += v
    for k, v in dictionary.items():
        print(f'\t\t{k}:\t\t\t {v}\t({100 * v / total:.1f}%)')

def draw_hist(data: list, title):
    _ = plt.hist(np.array(data)[:10000], bins='auto')  # arguments are passed to np.histogram
    plt.title(title)
    plt.show()


def extract_stats(data: list, file_path: None):

    questions_lengths = []
    answers_lengths = []
    number_of_answers = []
    context_lengths = []
    questions_count = 0
    impossible_q_count = 0
    answers_count = 0
    impossible_a_count = 0
    impossible_article = 0
    max_len = 0
    translated = 0
    non_translated = 0
    q_over_100 = 0
    a_over_100 = 0
    start_loc = []
    end_loc = []

    for paragraph in tqdm(data):
        has_impossible = False

        if 'translated' in paragraph and paragraph['translated']:
            translated += 1
        else:
            non_translated += 1

        context = paragraph['context']

        context_words = context.split(' ')
        context_lengths.append(len(context_words))

        question = paragraph['question']
        q_over_100 += 1 if len(question) > 100 else 0

        question_words = question.split(' ')
        questions_lengths.append(len(question_words))
        count_q_word(question)

        answers = paragraph['answers']

        impossible = len(answers['text']) == 0
        if impossible:
            impossible_q_count += 1
            has_impossible = True

        number_of_answers.append(len(answers['text']))
        for ans_text, start_index in zip(answers['text'], answers['answer_start']):
            a_over_100 += 1 if len(ans_text) > 100 else 0
            questions_count += 1
            ans_text_words = ans_text.split(' ')
            max_len = max(max_len, len(ans_text))
            answers_lengths.append(len(ans_text_words))
            answers_count += 1
            start_loc.append(start_index / len(context))
            end_loc.append((start_index + len(ans_text)) / len(context))
            if context[start_index:start_index + len(ans_text)] != ans_text:
                print("error")
    values, counts = np.unique(questions_first_words, return_counts=True)
    # print(f'longest answer: {max_len}')
    # print(f'q_over_100: {q_over_100}')
    # print(f'a_over_100: {a_over_100}')
    #
    # print(f'translated: {translated} not translated: {non_translated} ({translated * 100/ (translated + non_translated):.2f}%)')
    if file_path is not None:
        print(f'Statistics for file {file_path}')
    print(f'Number of questions: {questions_count}')
    print(f'Total articles: {len(data)}')
    print(f'Articles with impossible questions: {impossible_article}')
    print(f'Number of impossible questions: {impossible_q_count} ({100 * impossible_q_count / questions_count:.1f}%)')
    print(f'Number of possible questions: {questions_count - impossible_q_count} ({100 * (questions_count - impossible_q_count) / questions_count:.1f}%)')
    if opt.distribution:
        print(f'Question word distribution:')
        print_dict(questions_first_words, True)
    if opt.draw:
        draw_hist(questions_lengths, 'Questions Length')
        draw_hist(answers_lengths, 'Answers Length')
        draw_hist(number_of_answers, 'Number of answers per question')
        draw_hist(start_loc, "start location")
        draw_hist(end_loc, "end location")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('--match', type=str)
    parser.add_argument('--draw', action='store_true')
    parser.add_argument('--distribution', action='store_true')

    opt = parser.parse_args()

    if os.path.isdir(opt.input_path):
        files = os.listdir(opt.input_path)
        if opt.match:
            pattern = opt.match
            files = fnmatch.filter(os.listdir(opt.input_path), opt.match)

        files = [os.path.join(opt.input_path, f) for f in files]

    else:
        files = [opt.input_path]

    for file_path in files:

        with open(file_path) as json_file:
            full_doc = json.load(json_file)
            data = full_doc['data']

        extract_stats(data)

