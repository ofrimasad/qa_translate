""" Official evaluation script for v1.1 of the SQuAD dataset. """

import argparse
import json
import re
import string
import sys
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text) # r"\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b"

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def str_intersection_size(a: str, b: str):
    if len(a) > len(b):
        small = b
        big = a
    else:
        small = a
        big = b

    if small in big:
        return len(small)

    for span in range(len(small), len(small) // 2, -1):
        if small[:span] == big[-span:]:
            return span
        if small[-span:] == big[:span]:
            return span

    return 0


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def hebrew_bi_normalize(prediction, ground_truth):
    if len(prediction) == 0 or len(ground_truth) == 0:
        return prediction, ground_truth

    first_word_pred = prediction.split()[0]
    first_word_gt = ground_truth.split()[0]

    if first_word_gt == first_word_pred:
        return prediction, ground_truth

    short_len = min(len(first_word_pred), len(first_word_gt))
    if first_word_gt[-short_len:] == first_word_pred[-short_len:]:
        diff = len(first_word_pred) - len(first_word_gt)
        if diff > 0:
            return prediction[diff:], ground_truth
        else:
            return prediction, ground_truth[-diff:]

    return prediction, ground_truth


def f1_score_he(prediction, ground_truth):
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    prediction, ground_truth = hebrew_bi_normalize(prediction, ground_truth)

    num_same = str_intersection_size(prediction, ground_truth)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1

def exact_match_score_he(prediction, ground_truth):
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    prediction, ground_truth = hebrew_bi_normalize(prediction, ground_truth)
    return prediction == ground_truth

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def plot_cdf(values, x_label="F1 score"):

    hist, x = np.histogram(np.array(values), bins=100)
    y = np.cumsum(hist[::-1] / len(values))[::-1]
    plt.bar(x[1:-1], y[:-1] * 100, width=0.005)

    plt.xlabel(x_label)
    plt.ylabel("% of data over score")
    plt.title(f'Distribution of {x_label}')
    plt.text(0.8, 90, f'mean: {np.array(values).mean():.2f}')
    plt.text(0.8, 85, f'median: {np.median(np.array(values)):.2f}')
    plt.text(0.8, 80, f'>0: {100 * y[1]:.1f}%')
    plt.text(0.8, 75, f'=1: {100 * y[-1]:.1f}%')

    plt.show()


def evaluate(dataset, predictions, plot_dist=False):
    f1 = f1_he = exact_match = exact_match_he = total = 0
    f1_list, f1_he_list = [], []
    for article in dataset:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                total += 1
                if qa["id"] not in predictions:
                    message = "Unanswered question " + qa["id"] + " will receive score 0."
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x["text"], qa["answers"]))
                prediction = predictions[qa["id"]]
                exact_match += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
                _f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
                f1_list.append(_f1)
                f1 += _f1
                exact_match_he += metric_max_over_ground_truths(exact_match_score_he, prediction, ground_truths)
                _f1_he = metric_max_over_ground_truths(f1_score_he, prediction, ground_truths)
                f1_he_list.append(_f1_he)
                f1_he += _f1_he

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    exact_match_he = 100.0 * exact_match_he / total
    f1_he = 100.0 * f1_he / total

    if plot_dist:
        plot_cdf(f1_list)
        plot_cdf(f1_he_list, "F1' score")

    return {"exact_match": exact_match, "f1": f1, "exact_match_he": exact_match_he, "f1_he": f1_he}


if __name__ == "__main__":

    expected_version = "1.1"
    parser = argparse.ArgumentParser(description="Evaluation for SQuAD " + expected_version)
    parser.add_argument("dataset_file", help="Dataset file")
    parser.add_argument("prediction_file", help="Prediction File")
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if dataset_json["version"] != expected_version:
            print(
                "Evaluation expects v-" + expected_version + ", but got dataset with v-" + dataset_json["version"],
                file=sys.stderr,
            )
        dataset = dataset_json["data"]
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(dataset, predictions)))
