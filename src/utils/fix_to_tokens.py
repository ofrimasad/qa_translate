import argparse
import json

from tqdm import tqdm


if __name__ == "__main__":

    """
    fix a dataset so that each answer will start after a space
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, required=True)
    parser.add_argument('-o', '--output_path', type=str, required=True)

    opt = parser.parse_args()

    with open(opt.input_path) as json_file:
        full_doc = json.load(json_file)
        data = full_doc['data']

    for paragraph in tqdm(data):
        start = paragraph['answers']['answer_start'][0]
        if start > 0:
            if paragraph['context'][start - 1] != ' ':
                ans = paragraph['answers']['text'][0]
                end = start + len(ans)


                index_last_space = paragraph['context'][:start].rfind(" ")
                new_ans = paragraph['context'][index_last_space + 1: end]

                paragraph['answers']['answer_start'][0] = index_last_space + 1
                ans = paragraph['answers']['text'][0] = new_ans

    with open(opt.output_path, 'w') as json_out:
        full_doc['data'] = data
        json.dump(full_doc, json_out, ensure_ascii=False)
        print(f'file saved: {opt.output_path}')