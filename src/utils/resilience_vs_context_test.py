import argparse
import json
import random
import re
from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from languages import LANGUAGES
from languages.english import English
from services.google_translate import GoogleTranslate
from utils.translation_utils import SentenceSpliter


def index_to_sentence_index(index: int, sentences: list):
    """
    :param index: original index
    :param sentences: list of sentences
    :return: index of containing sentence, index in the containing sentence
    """
    total = 0
    total_prev = 0
    for i, s in enumerate(sentences):
        total += len(s) + 1
        if index < total:
            return i, index - total_prev
        total_prev = total

class CosineCorr:

    def __init__(self, model_name_or_path: str):
        self.model = SentenceTransformer(model_name_or_path)

    def __call__(self, a: str, list_b: List[str]):

        a = self.model.encode([a], show_progress_bar=False, batch_size=32, convert_to_tensor=True)
        b = self.model.encode(list_b, show_progress_bar=False, batch_size=32, convert_to_tensor=True)

        return util.cos_sim(a, b)


def clean_markers(cont: str, marker: str) -> str:
    return cont.replace(marker, "")

def format_line(label: str, lst: Union[List[str], np.ndarray]) -> str:
    if isinstance(lst, list):
        return f"{label},{','.join(lst)}\n"
    else:
        return f"{label},{','.join([str(x) for x in lst.tolist()])}\n"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_json', type=str)
    parser.add_argument('language_sym', type=str)

    opt = parser.parse_args()
    debug = False
    MARKERS = ["\n" , "[34456]", "<p>", "<<H>>", "[P]", "##", "@@", "&&&", "$$", "</b>", "{3333}", "(%%%)", "__", ">><<", "*~~*", "[^^]"]

    survived = np.zeros(shape=len(MARKERS))
    didnt_survived = np.zeros(shape=len(MARKERS))
    correlations = []
    paragraph_count = 0

    for k, v in opt.__dict__.items(): print(f'{k}:\t{v}')

    target = LANGUAGES[opt.language_sym]
    translator = GoogleTranslate(source=English, target=target)

    cosine_correlation = CosineCorr("bert-base-multilingual-cased")

    sentence_splitter = SentenceSpliter()

    with open(opt.input_json) as json_file:
        full_doc = json.load(json_file)
        data = full_doc['data']

    try:
        for ind, subject in enumerate(data[:5]):

            print(f'Part {ind + 1} / {len(data)}\n')

            paragraphs = subject['paragraphs']

            for paragraph in tqdm(paragraphs):
                paragraph_count += 1
                if paragraph_count > 1000:
                    break

                context = paragraph['context']
                sentences = sentence_splitter.sentence_split(context)
                sentences_translated = translator.translate_together(sentences)

                qas = paragraph['qas']
                qas = random.choices(qas, k=4)
                for qa in qas:
                    if 'is_impossible' in qa and qa['is_impossible']:
                        continue

                    answer = qa['answers'][0]
                    ans_text = answer['text']
                    answer_start = answer["answer_start"]
                    sentence_index, answer_start = index_to_sentence_index(answer_start, sentences)
                    answer_end = answer_start + len(ans_text)
                    sentence = sentences[sentence_index]
                    assert sentences[sentence_index][answer_start:answer_end] == ans_text

                    translated_with_markers_list = []
                    for marker_index, marker in enumerate(MARKERS):
                        pre = sentence[:answer_start] if answer_start > 0 else ""
                        post = sentence[answer_end:] if answer_end < len(sentence) - 1 else ""
                        sentence_with_markers = pre + marker + ans_text + marker + post
                        translated_with_markers = translator.translate(sentence_with_markers)
                        markers_after_translation = [_.start() for _ in re.finditer(re.escape(marker), translated_with_markers)]
                        if len(markers_after_translation) == 2:
                            survived[marker_index] += 1
                        else:
                            didnt_survived[marker_index] += 1

                        translated_with_markers_list.append(clean_markers(translated_with_markers, marker))

                    correlations.append(cosine_correlation(sentences_translated[sentence_index], translated_with_markers_list).cpu().numpy())
                    if debug:
                        print("=" * 80)
                        print(sentences_translated[sentence_index])
                        for tr, cor in zip(translated_with_markers_list, correlations[-1][0].tolist()):
                            print(f"{cor:.3f}: {tr}")

    finally:
        correlations = np.concatenate(correlations)
        correlations_mean = correlations.mean(axis=0)
        correlations_median = np.median(correlations, axis=0)
        correlations_std = correlations.std(axis=0)
        survival_rate = survived / (survived + didnt_survived)


        lines = [f"lang,{opt.language_sym}\n"]
        lines.append(format_line("markers", MARKERS))
        lines.append(format_line("survived", survived))
        lines.append(format_line("didnt_survived", didnt_survived))
        lines.append(format_line("survival_rate", survival_rate))
        lines.append(format_line("correlations_mean", correlations_mean))
        lines.append(format_line("correlations_std", correlations_std))
        lines.append(format_line("correlations_median", correlations_median))

        with open('/home/ofri/qa_translate/resilience_vs_context_test_summary.csv', "a") as summary:

            for l in lines:
                print(l)
                summary.write(l)
