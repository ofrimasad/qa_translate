import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.smart_match import CorrelationMatcher, ModelMatcher

BASE_DIR = '/home/ofri/qa_translate/matcher_eval/'

def plot_precision_recall(results: np.ndarray, model_name):
    scores, correct, possible = np.split(results, 3, axis=1)

    thresholds = np.arange(0.00, 1.01, 0.01)
    precision = np.zeros(101)
    recall = np.zeros(101)
    em = np.zeros(101)

    for i, threshold in enumerate(thresholds):

        tp = np.count_nonzero(np.logical_and(scores >= threshold, possible == 1))
        fp = np.count_nonzero(np.logical_and(scores >= threshold, possible == 0))
        positive = np.count_nonzero(possible)

        precision[i] = tp / (tp + fp + 1e-6)
        recall[i] = tp / (positive + 1e-6)

        em_count = np.count_nonzero(np.logical_and(scores >= threshold, correct == 1))
        em[i] = em_count / (tp + fp + 1e-6)

    recall = recall[::-1]
    precision = precision[::-1]

    # precision[0] = precision[1]
    em[-1] = em[-2]

    plt.plot(recall, precision)
    plt.title('Precision - Recall')
    plt.suptitle(model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1.1)
    if precision[-2] - precision[-1] > 0.1:
        plt.annotate(f'th:{thresholds[2]:.2f}\np:{precision[-2]:.2f} r: {recall[-2]:.2f}', xy=(recall[-2], precision[-2]))
    plt.show()

    plt.plot(thresholds, em)
    plt.title('EM (Possible)')
    plt.suptitle(model_name)
    plt.xlabel('Thresh')
    plt.ylabel('EM')
    plt.xlim(0, 1)
    plt.ylim(0, 1.1)
    plt.show()

    save_path = BASE_DIR + model_name + '_PRC.csv'
    np.savetxt(save_path, np.stack([recall, precision]).transpose(), delimiter=',', fmt='%f')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('val_data_json', type=str)
    parser.add_argument('--max_data_size', type=int, default=1000)
    parser.add_argument('--correlation', action='store_true')

    opt = parser.parse_args()

    if opt.correlation:
        matcher = CorrelationMatcher(model_name_or_path='bert-base-multilingual-cased') # 'onlplab/alephbert-base'
    else:
        matcher = ModelMatcher(model_name_or_path=opt.model) # '/home/ofri/qa_translate/exp_xquad/train_ss_dee'

    with open(opt.val_data_json) as json_file:
        full_doc = json.load(json_file)
        data = full_doc['data']

    scores = []
    correct = []
    possible = []
    if len(data) > opt.max_data_size:
        data = data[:opt.max_data_size]

    for qa in tqdm(data):

        context = qa['context']

        question = qa['question']
        answers = qa['answers']

        prediction, score = matcher.match(context, question)
        scores.append(score)
        if len(answers['text']) > 0:
            target = answers['text'][0]
            correct.append(1 if target == prediction else 0)
            possible.append(1)
        else:
            correct.append(0)
            possible.append(0)

    scores = np.array(scores)
    correct = np.array(correct)
    possible = np.array(possible)

    results = np.stack([scores, correct, possible]).transpose()
    save_path = BASE_DIR + opt.model.split("/")[-1] + '_raw.csv'
    np.savetxt(save_path, results, delimiter=',', fmt='%f')

    plot_precision_recall(results, opt.model.split("/")[-1])
