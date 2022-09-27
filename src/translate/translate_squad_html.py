import argparse
import json
import re
from collections import namedtuple

import wikipediaapi
from tqdm import tqdm

from languages.hebrew_html import HebrewHTML
from services.google_translate_paied import GoogleTranslateP
from src.languages.english import English
from src.languages.hebrew import Hebrew
from src.services.google_translate import GoogleTranslate
from src.utils.smart_match import CorrelationMatcher
from src.utils.utils import DictionaryLink, TextList, get_git_revision_short_hash, HtmlTagger

wiki_wiki = wikipediaapi.Wikipedia('en')

SEP = '34456'


class Stats:
    def __init__(self):
        self.orig_multiple_indices = 0
        self.orig_single_index = 0
        self.trans_multiple_indices = 0
        self.trans_single_index = 0
        self.same_num_of_sections = 0
        self.different_num_of_sections = 0
        self.lost_in_trans = 0
        self.not_lost_in_trans = 0
        self.found_in_wiki = 0
        self.not_found_in_wiki = 0
        self.matched_and_replaced = 0
        self.could_not_replace = 0

    def __str__(self):
        _str = ''
        _str += f'\noriginal - multiple occurrences of answer: {self.orig_multiple_indices} ({100 * self.orig_multiple_indices / (self.orig_multiple_indices + self.orig_single_index):.1f}%)'
        _str += f'\noriginal - single occurrence of answer: {self.orig_single_index} ({100 * self.orig_single_index / (self.orig_multiple_indices + self.orig_single_index):.1f}%)'
        _str += f'\ntranslated - multiple occurrences of answer: {self.trans_multiple_indices} ({100 * self.trans_multiple_indices / (self.trans_multiple_indices + self.trans_single_index):.1f}%)'
        _str += f'\ntranslated - single occurrence of answer: {self.trans_single_index} ({100 * self.trans_single_index / (self.trans_multiple_indices + self.trans_single_index):.1f}%)'
        _str += f'\nsame number of sub contexts: {self.same_num_of_sections} ({100 * self.same_num_of_sections / (self.same_num_of_sections + self.different_num_of_sections):.1f}%)'
        _str += f'\ndifferent number of sub contexts: {self.different_num_of_sections} ({100 * self.different_num_of_sections / (self.same_num_of_sections + self.different_num_of_sections):.1f}%)'
        _str += f'\nanswer lost in trans: {self.lost_in_trans} ({100 * self.lost_in_trans / (self.lost_in_trans + self.not_lost_in_trans):.1f}%)'
        _str += f'\nanswer not lost in trans: {self.not_lost_in_trans} ({100 * self.not_lost_in_trans / (self.lost_in_trans + self.not_lost_in_trans):.1f}%)'
        if self.matched_and_replaced + self.could_not_replace > 0:
            _str += f'\nanswer replaced: {self.matched_and_replaced} ({100 * self.matched_and_replaced / (self.matched_and_replaced + self.could_not_replace):.1f}%)'
            _str += f'\nanswer not replaced: {self.could_not_replace} ({100 * self.could_not_replace / (self.matched_and_replaced + self.could_not_replace):.1f}%)'

        return _str


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_json', type=str, required=True)
    parser.add_argument('-o', '--output_json', type=str, required=True)
    parser.add_argument('-r', '--readable', action='store_true', help='readable json output format')
    parser.add_argument('--no_markers', action='store_true', help='do not use markers to split context')
    parser.add_argument('--replace', action='store_true', help='replace answers missing from context')

    opt = parser.parse_args()

    translator = GoogleTranslateP(source=English, target=HebrewHTML)

    with open(opt.input_json) as json_file:
        full_doc = json.load(json_file)
        data = full_doc['data']

    try:
        for ind, subject in enumerate(data):

            print(f'Part {ind + 1} / {len(data)}\n')

            paragraphs = subject['paragraphs']

            for paragraph in tqdm(paragraphs):

                if 'translated' in paragraph and paragraph['translated']:
                    continue

                text_list = TextList()
                text_list.append(subject, 'title')

                context = paragraph['context']
                tagger = HtmlTagger(context)

                qas = paragraph['qas']
                for qa in qas:
                    text_list.append(qa, 'question')
                    if 'plausible_answers' in qa:
                        answers = qa['plausible_answers']
                    else:
                        answers = qa['answers']

                    possible = False
                    for ans in answers:
                        ans_text = ans['text']
                        index_start = ans['answer_start']
                        index_end = index_start + len(ans_text)
                        context, start_tag, end_tag = tagger.insert_tags(context, index_start, index_end, with_shift=True)
                        ans['text'] = (start_tag, end_tag)

                paragraph['context'] = context
                text_list.append(paragraph, 'context')
                translated_text_list = translator.translate_together(text_list.texts)

                original_context = paragraph['context']
                translated_context = translated_text_list[-1]

                for text, link, translated_text in zip(text_list.texts, text_list.links, translated_text_list):
                    link.set(translated_text)

                clean_translated_context = tagger.clean(translated_context)
                translated_context = tagger.fix_tags(translated_context)
                for qa in qas:
                    if 'plausible_answers' in qa:
                        answers = qa['plausible_answers']
                    else:
                        answers = qa['answers']

                    for ans in answers:
                        start_tag, end_tag = ans['text']
                        index_start = tagger.get_text_unshift(translated_context, translated_context.find(start_tag))
                        index_end = tagger.get_text_unshift(translated_context, translated_context.find(end_tag))
                        ans['text'] = clean_translated_context[index_start:index_end]
                        ans['answer_start'] = index_start
                        assert len(ans['text']) > 0
                        assert ans['answer_start'] >= 0


                paragraph['translated'] = True
                paragraph['context'] = clean_translated_context

        # second pass - clean answers with None answer_start
        for d in tqdm(data):
            paragraphs = d['paragraphs']
            new_paragraphs = []
            for paragraph in paragraphs:
                qas = paragraph['qas']
                new_qas = []
                for qa in qas:
                    if 'plausible_answers' in qa:
                        answers = qa['plausible_answers']
                    else:
                        answers = qa['answers']

                    new_answers = []
                    for ans in answers:
                        if ans["answer_start"] >= 0:
                            new_answers.append(ans)

                    if 'plausible_answers' in qa:
                        qa['plausible_answers'] = new_answers
                    else:
                        qa['answers'] = new_answers

                    if len(new_answers) > 0:
                        new_qas.append(qa)
                paragraph['qas'] = new_qas
                if len(new_qas) > 0:
                    new_paragraphs.append(paragraph)
            d['paragraphs'] = new_paragraphs

    finally:
        print('Saving to file')
        with open(opt.output_json, 'w') as json_out:
            full_doc['data'] = data
            json.dump(full_doc, json_out, ensure_ascii=False, indent=3 if opt.readable else None)
            print(f'file saved: {opt.output_json}')
        with open(opt.output_json.replace('json', 'txt'), 'w') as text_out:
            text_out.write("\n".join(f'{o[0]}: {o[1]}' for o in opt.__dict__.items()))
            text_out.write(f'\ncommit: {get_git_revision_short_hash()}')
            print(f'file saved: {opt.output_json.replace("json", "txt")}')