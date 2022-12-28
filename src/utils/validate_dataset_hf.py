import argparse
import json
import os

from tqdm import tqdm


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)

    opt = parser.parse_args()


    if opt.input_path.endswith('.json'):
        files = [opt.input_path]
    else:
        files = os.listdir(opt.input_path)
        files = [os.path.join(opt.input_path, f) for f in sorted(files) if f.endswith('.json') and "hf" in f and "train" in f]

    for file in files:
        with open(file) as json_file:
            full_doc = json.load(json_file)
            data = full_doc['data']

        invalid_ans_count = 0
        total_ans_count = 0

        for paragraph in tqdm(data):
            context = paragraph['context']
            question = paragraph['question']
            answers = paragraph['answers']

            for ans_text,start  in zip(answers['text'], answers["answer_start"]):
                ans_text = ans_text.strip('.')
                end = start + len(ans_text)
                if context is None:
                    invalid_ans_count += 1
                    continue
                if context[start:end] != ans_text:
                    invalid_ans_count += 1
                total_ans_count += 1

        print(f'{file}: errors {invalid_ans_count} / {total_ans_count} ({invalid_ans_count / total_ans_count * 100}%)')