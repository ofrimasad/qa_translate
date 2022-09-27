import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.utils.smart_match import CorrelationMatcher

BASE_DIR = '/home/ofri/qa_translate/matcher_eval/'


def plot_precision_recall(results: np.ndarray, model_name):
    scores, tp, impossibles = np.split(results, 3, axis=1)

    x_s = np.arange(0.01, 1.01, 0.01)
    y_s = np.zeros_like(x_s)

    for i, x in enumerate(x_s):
        threshold = x
        success = np.logical_and(np.logical_and(scores >= threshold, impossibles == 0), tp == 1)
        empty_success = np.logical_and(scores < threshold, impossibles == 1)
        total_success = np.logical_or(success, empty_success).astype(np.int)
        count = np.count_nonzero(total_success)
        y_s[i] = count / len(scores)
    plt.plot(x_s, y_s)
    plt.title('Precision - Recall')
    plt.suptitle(model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
    save_path = BASE_DIR + model_name + '_PRC.csv'
    np.savetxt(save_path, np.stack([x_s, y_s]).transpose(), delimiter=',', fmt='%f')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('val_data_json', type=str)

    opt = parser.parse_args()

    matcher = CorrelationMatcher(model_name_or_path=opt.model) # 'onlplab/alephbert-base'

    with open(opt.val_data_json) as json_file:
        full_doc = json.load(json_file)
        data = full_doc['data']

    scores = []
    tp = []
    impossibles = []

    for qa in tqdm(data):

        context = qa['context']

        question = qa['question']
        impossible = 0 # qa['is_impossible']

        if 'plausible_answers' in qa:
            answers = qa['plausible_answers']
        else:
            answers = qa['answers']

        target = answers['text'][0]

        prediction, score = matcher.match(context, question)

        tp.append(1 if target == prediction else 0)
        scores.append(score)
        impossibles.append(impossible)

        # print(f'[{1 if target == prediction else 0}]\t{target}\t\t\ {prediction}')

    scores = np.array(scores)
    tp = np.array(tp)
    impossibles = np.array(impossibles).astype(np.int)

    results = np.stack([scores, tp, impossibles]).transpose()
    save_path = BASE_DIR + opt.model.replace('/', '_') + '_raw.csv'
    np.savetxt(save_path, results, delimiter=',', fmt='%f')

    plot_precision_recall(results, opt.model.replace('/', '_'))
