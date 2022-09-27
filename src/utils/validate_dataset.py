import argparse
import json
import os

from tqdm import tqdm


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, required=True)

    opt = parser.parse_args()


    if opt.input_path.endswith('.json'):
        files = [opt.input_path]
    else:
        files = os.listdir(opt.input_path)
        files = [os.path.join(opt.input_path, f) for f in files if f.endswith('.json')]

    for file in files:
        with open(file) as json_file:
            full_doc = json.load(json_file)
            data = full_doc['data']

        invalid_ans_count = 0
        total_ans_count = 0

        for subject in tqdm(data):

            paragraphs = subject['paragraphs']
            for paragraph in paragraphs:

                context = paragraph['context']
                qas = paragraph['qas']
                for qa in qas:
                    question = qa['question']

                    if 'plausible_answers' in qa:
                        answers = qa['plausible_answers']
                    else:
                        answers = qa['answers']

                    for ans in answers:
                        ans_text = ans['text']
                        ans_text = ans_text.strip('.')
                        start = ans["answer_start"]
                        end = start + len(ans_text)
                        if context is None:
                            invalid_ans_count += 1
                            continue
                        if context[start:end] != ans_text:
                            invalid_ans_count += 1
                        total_ans_count += 1

        print(f'{file}: errors {invalid_ans_count} / {total_ans_count}')