import argparse
import json
import re

import wikipediaapi
from tqdm import tqdm

from languages import LANGUAGES
from src.languages.english import English
from src.services.google_translate import GoogleTranslate
from src.utils.smart_match import CorrelationMatcher
from src.utils.utils import DictionaryLink, TextList, get_git_revision_short_hash

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


def add_markers(paragraph: dict):
    paragraph['context'] = paragraph['context'].replace('. ', f'. [{SEP}] ')


def fix_sep(context: str):
    return re.sub('3,?\.?4,?\.?4,?\.?5,?\.?6', SEP, context, count=100)


def clean_translated_context(context: str):
    context = fix_sep(context)
    return context.replace(f'[{SEP}]', '').replace(f'[ {SEP}]', '').replace(f'[{SEP} ]', '').replace(f'{SEP}]', '').replace(f'[{SEP}', '').replace(SEP, '')


def clean_translated_sub_context(sub_context: str):
    return sub_context.strip().strip('[').strip(']')


def align_indices(original_context: str, translated_context: str, original_text: str, translated_text: str,
                  link: DictionaryLink, matcher: CorrelationMatcher = None, match_thresh: float = 0.6, stats: Stats = None):

    full_translated_context = None
    try:
        using_separetor = SEP in original_context

        original_text = original_text.strip().strip('.')
        translated_text = translated_text.strip().strip('.')

        link.set(translated_text)

        translated_context = fix_sep(translated_context)

        original_context = original_context.replace(f' [{SEP}] ', ' ')
        original_start_index = link.object["answer_start"]

        if using_separetor:
            sentence_index = original_context[0:original_start_index].count('. ')
            original_sub_context = original_context.split('. ')[sentence_index]
            translated_sub_context = translated_context.split(SEP)[sentence_index]

        if using_separetor and len(original_context.split('. ')) == len(translated_context.split(SEP)):
            # IF SENTENCE SPLITTING WORKED PROPERLY
            stats.same_num_of_sections += 1

            # clean context cnd sub_context from markers
            translated_sub_context = clean_translated_sub_context(translated_sub_context)
            translated_context = clean_translated_context(translated_context)

            # find the offset of the sub context in the context
            original_offset = original_context.find(original_sub_context)
            translated_offset = translated_context.find(translated_sub_context)

            # replace the context with the sub context (for the next phase)
            original_context = original_sub_context
            full_translated_context = translated_context
            translated_context = translated_sub_context

        else:
            stats.different_num_of_sections += 1
            original_offset = 0
            translated_offset = 0
            translated_context = clean_translated_context(translated_context)
            full_translated_context = translated_context

        original_start_indices = [_.start() for _ in re.finditer(re.escape(original_text), original_context)]
        original_start_index = link.object["answer_start"] - original_offset

        translated_start_indices = [_.start() for _ in re.finditer(re.escape(translated_text), translated_context)]

        if len(original_start_indices) > 1:
            stats.orig_multiple_indices += 1
        if len(original_start_indices) == 1:
            stats.orig_single_index += 1
        if len(translated_start_indices) > 1:
            stats.trans_multiple_indices += 1
        if len(translated_start_indices) == 1:
            stats.trans_single_index += 1

        new_translated_context = None
        if original_start_index not in original_start_indices:
            # this was an impossible question - leave it that way
            if len(translated_start_indices) == 1:
                link.object["answer_start"] = translated_start_indices[0] + translated_offset
                stats.not_lost_in_trans += 1
            else:
                link.object["answer_start"] = -1
                stats.lost_in_trans += 1
        elif len(translated_start_indices) == 0:
            # translation does not include the answer
            if matcher is not None:
                new_translated_context = replace(link, match_thresh, matcher, stats, translated_context, translated_text, translated_offset)
                full_translated_context = full_translated_context.replace(translated_context, new_translated_context)
            else:
                link.object["answer_start"] = -1
                stats.lost_in_trans += 1
        else:
            occurrence_index = original_start_indices.index(original_start_index)
            if occurrence_index < len(translated_start_indices):
                # take the occurrence of the answer by occurrence index
                stats.not_lost_in_trans += 1
                link.object["answer_start"] = translated_start_indices[occurrence_index] + translated_offset
            else:
                # could not find the occurrence
                if matcher is not None:
                    new_translated_context = replace(link, match_thresh, matcher, stats, translated_context, translated_text, translated_offset)
                    full_translated_context = full_translated_context.replace(translated_context, new_translated_context)
                else:
                    link.object["answer_start"] = -1
                    stats.lost_in_trans += 1

    except Exception as e:
        link.object["answer_start"] = -1

    return full_translated_context


def replace(link: DictionaryLink, match_thresh: float, matcher: CorrelationMatcher, stats: Stats, translated_context: str,
            translated_text: str, translated_offset: int):
    found, score = matcher.match(translated_context, translated_text)
    if score > match_thresh:
        translated_start_indices = [_.start() for _ in re.finditer(re.escape(found), translated_context)]
        index = translated_start_indices[0]
        translated_context_new = translated_context[:index] + translated_context[index:].replace(found, link.object['text'], 1)
        link.object["answer_start"] = translated_start_indices[0] + translated_offset
        stats.matched_and_replaced += 1
        stats.not_lost_in_trans += 1
        link.object['replaced'] = True
        return translated_context_new
    else:
        link.object["answer_start"] = -1
        stats.lost_in_trans += 1
        stats.could_not_replace += 1
        return translated_context


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_json', type=str)
    parser.add_argument('language_sym', type=str)
    parser.add_argument('-r', '--readable', action='store_true', help='readable json output format')
    parser.add_argument('-w', '--wiki', action='store_true', help='fetch wiki context')
    parser.add_argument('--no_markers', action='store_true', help='do not use markers to split context')
    parser.add_argument('--replace', action='store_true', help='replace answers missing from context')
    parser.add_argument('--skip_impossible', action='store_true', help='skip questions which are impossible')

    opt = parser.parse_args()
    stats = Stats()

    target = LANGUAGES[opt.language_sym]
    output_json = opt.input_json.replace('.json', f'_{target.symbol}_aug.json')
    translator = GoogleTranslate(source=English, target=target)
    matcher = CorrelationMatcher(model_name_or_path='bert-base-multilingual-cased') if opt.replace else None

    with open(opt.input_json) as json_file:
        full_doc = json.load(json_file)
        data = full_doc['data']

    new_data = []

    try:
        for ind, subject in enumerate(data):

            print(f'Part {ind + 1} / {len(data)}\n')

            paragraphs = subject['paragraphs']
            new_paragraphs = []

            if opt.wiki:
                page_py = wiki_wiki.page(subject['title'])

            for paragraph in tqdm(paragraphs):

                # if 'translated' in paragraph and paragraph['translated']:
                #     continue
                # if not paragraph['context'].startswith('City and Guilds College was founded'):
                #     continue

                text_list = TextList()
                text_list.append(subject, 'title')
                if not opt.no_markers:
                    add_markers(paragraph)
                text_list.append(paragraph, 'context')

                qas = paragraph['qas']

                skip_all = True
                for qa in qas:
                    if qa['is_impossible'] and opt.skip_impossible:
                        continue

                    skip_all = False
                    text_list.append(qa, 'question')
                    if 'plausible_answers' in qa:
                        answers = qa['plausible_answers']
                    else:
                        answers = qa['answers']

                    for ans in answers:
                        ans_text = ans['text']
                        text_list.append(ans, 'text')

                if skip_all:
                    qas = []
                    continue

                translated_text_list = translator.translate_together(text_list.texts)

                original_context = paragraph['context']


                translated_title = translated_text_list[0]
                translated_context = translated_text_list[1]

                qas_list = []
                qa = {'answers':[]}
                for i in range(2, len(translated_text_list)):
                    text = text_list.texts[i]
                    link = text_list.links[i]
                    translated_text = translated_text_list[i]

                    if link.label == 'text':  # this is an answer text
                        # this is an answer text
                        new_translated_context = align_indices(original_context, translated_context, text, translated_text, link, matcher=matcher, stats=stats)
                        qa['answers'].append(link.object)
                    else:
                        if len(qas_list) > 0:
                            new_paragraphs.append({'context': new_translated_context, 'qas': qas_list})
                        #     qas_list = []
                        #     qa = {'answers': []}
                        qa = {'answers': []}
                        qa['question'] = translated_text
                        qa['id'] = link.object['id']
                        qas_list= [qa]

                # append th last qa
                if len(qas_list) > 0:
                    new_paragraphs.append({'context': new_translated_context, 'qas': qas_list})

            new_data.append({'paragraphs': new_paragraphs})
            # time.sleep(30)

        # second pass - clean answers with None answer_start
        for d in tqdm(new_data):
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
    except Exception as e:
        print(e)
    finally:
        print('Saving to file')
        with open(output_json, 'w') as json_out:
            full_doc['data'] = new_data
            json.dump(full_doc, json_out, ensure_ascii=False, indent=3 if opt.readable else None)
            print(f'file saved: {output_json}')
        with open(output_json.replace('json', 'txt'), 'w') as text_out:
            text_out.write("\n".join(f'{o[0]}: {o[1]}' for o in opt.__dict__.items()))
            text_out.write(f'\ncommit: {get_git_revision_short_hash()}')
            text_out.write('\n\n===== stats =====\n')
            text_out.write(str(stats))
            print(f'file saved: {output_json.replace("json", "txt")}')