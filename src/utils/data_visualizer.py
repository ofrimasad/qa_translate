import json
import random

import argparse
import http.server
import socketserver


def insert(source_str, insert_str, pos):
    return source_str[:pos] + insert_str + source_str[pos:]


def get_html_head():
    all_lines = []
    all_lines.append('<!DOCTYPE html>')
    all_lines.append('<html>')
    all_lines.append('<head><meta charset="UTF-8"></head>')
    all_lines.append(
        "<style>table, th, td {border:1px solid black;} meter{width:90%;height:30px; margin: -7px;} table.center {margin-left: auto; margin-right: auto; margin-top: 60px;}</style>")
    all_lines.append('<body>')
    return all_lines


def get_html_footer():
    all_lines = []
    all_lines.append('</body>')
    all_lines.append('</html>')
    return all_lines


def get_html_table(context, question, impossible, answers_ind, lang, direction):
    all_lines = []
    all_lines.append('<table style="width:60%;" class="center">')
    all_lines.append(f'<tr><td dir="{direction}" lang="{lang}" colspan="2">{question}</td></tr>')
    all_lines.append(f'<tr><td colspan="2">{"<mark>Impossible</mark>" if impossible else ""}</td></tr>')
    for ans, index in answers_ind:
        all_lines.append(f'<tr><td>start index: {index}</td><td dir="{direction}" lang="{lang}">{ans}</td></tr>')

    used = []

    for ans, index in answers_ind[::-1]:
        if index not in used:
            context = insert(context, '</mark>', pos=index + len(ans))
            context = insert(context, '<mark>', pos=index)
            used.append(index)

    all_lines.append(f'<tr><td colspan="2" dir="{direction}" lang="{lang}">{context}</td></tr></table>')

    return all_lines


class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):

    def do_GET(self):

        all_lines = []
        all_lines.extend(get_html_head())

        paragraphs_i = random.randint(0, len(data) - 1)
        paragraphs = data[paragraphs_i]['paragraphs']
        paragraph_i = random.randint(0, len(paragraphs) - 1)
        paragraph = paragraphs[paragraph_i]
        context = paragraph['context']
        qas = paragraph['qas']
        qa_i = random.randint(0, len(qas) - 1)
        qa = qas[qa_i]
        question = qa['question']

        if 'plausible_answers' in qa:
            answers = qa['plausible_answers']
        else:
            answers = qa['answers']

        impossible = qa["is_impossible"]

        answers_ind = []
        for ans in answers:
            ans_text = ans['text']
            index = ans['answer_start']
            answers_ind.append((ans_text, index))

        all_lines.extend(get_html_table(context, question, impossible, answers_ind, 'he', 'rtl'))

        if data_en is not None:
            paragraphs_en = data_en[paragraphs_i]['paragraphs']
            paragraph_en = paragraphs_en[paragraph_i]
            context_en = paragraph_en['context']
            qas_en = paragraph_en['qas']
            qa_en = qas_en[qa_i]
            question_en = qa_en['question']
            if 'plausible_answers' in qa:
                answers_en = qa_en['plausible_answers']
            else:
                answers_en = qa_en['answers']

            answers_ind_en = []
            for ans in answers_en:
                ans_text = ans['text']
                index = ans['answer_start']
                answers_ind_en.append((ans_text, index))

            all_lines.extend(get_html_table(context_en, question_en, impossible, answers_ind_en, 'en', 'ltr'))

        all_lines.extend(get_html_footer())

        html = "\n".join(all_lines)

        self.send_response(200)
        self.send_header("Content-type", "text/html")
        # self.send_header("Content-length", str(len(html)))

        self.end_headers()
        self.wfile.write(html.encode('UTF-8'))


def filter_replaced_only(data: list):
    for d in data:
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
                    if 'replaced' in ans and ans['replaced']:
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
    return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_json', type=str, required=True)
    parser.add_argument('-en', '--english_version', type=str)
    parser.add_argument('-p', '--port', type=int, default=8000)
    parser.add_argument('--replaced_only', action='store_true', help='show only answers which were replaced')
    opt = parser.parse_args()

    with open(opt.input_json) as json_file:
        full_doc = json.load(json_file)
        data = full_doc['data']

    if opt.replaced_only:
        data = filter_replaced_only(data)

    data_en = None
    if opt.english_version is not None:
        with open(opt.english_version) as json_file:
            full_doc = json.load(json_file)
            data_en = full_doc['data']

    handler = MyHttpRequestHandler

    with socketserver.TCPServer(("", opt.port), handler) as httpd:
        print("Server started at localhost:" + str(opt.port))
        httpd.serve_forever()

    httpd.server_close()
