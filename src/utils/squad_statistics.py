import argparse
import json

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_json', type=str, required=True)

    opt = parser.parse_args()

    with open(opt.input_json) as json_file:
        full_doc = json.load(json_file)
        data = full_doc['data']

    questions_lengths = []
    answers_lengths = []
    number_of_answers = []
    context_lengths = []
    questions_count = 0
    impossible_q_count = 0
    answers_count = 0
    impossible_a_count = 0
    impossible_article = 0

    translated = 0
    non_translated = 0
    for subject in tqdm(data):
        has_impossible = False

        paragraphs = subject['paragraphs']
        for paragraph in paragraphs:

            if 'translated' in paragraph and paragraph['translated']:
                translated += 1
            else:
                non_translated += 1

            context = paragraph['context']
            context_words = context.split(' ')
            context_lengths.append(len(context_words))
            qas = paragraph['qas']
            for qa in qas:
                question = qa['question']
                question_words = question.split(' ')
                questions_lengths.append(len(question_words))
                count_q_word(question)

                questions_count += 1

                if 'plausible_answers' in qa:
                    answers = qa['plausible_answers']
                else:
                    answers = qa['answers']

                impossible = "is_impossible" in qa and qa["is_impossible"]
                if impossible:
                    impossible_q_count += 1
                    has_impossible = True

                number_of_answers.append(len(answers))
                for ans in answers:
                    ans_text = ans['text']
                    ans_text_words = ans_text.split(' ')
                    answers_lengths.append(len(ans_text_words))
                    answers_count += 1
                    if impossible:
                        impossible_a_count += 1
        if has_impossible:
            impossible_article += 1
    values, counts = np.unique(questions_first_words, return_counts=True)

    print(f'translated: {translated} not translated: {non_translated} ({translated * 100/ (translated + non_translated):.2f}%)')
    print(f'Statistics for file {opt.input_json}')
    print(f'Number of questions: {questions_count}')
    print(f'Total articles: {len(data)}')
    print(f'Articles with impossible questions: {impossible_article}')
    print(f'Number of impossible questions: {impossible_q_count} ({100 * impossible_q_count / questions_count:.1f}%)')
    print(f'Number of possible questions: {questions_count - impossible_q_count} ({100 * (questions_count - impossible_q_count) / questions_count:.1f}%)')
    print(f'Question word distribution:')
    print_dict(questions_first_words, True)
    draw_hist(questions_lengths, 'Questions Length')
    draw_hist(answers_lengths, 'Answers Length')
    draw_hist(number_of_answers, 'Number of answers per question')

