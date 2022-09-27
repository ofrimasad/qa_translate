

import argparse
import json
import os.path


import glob
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)

    opt = parser.parse_args()

    for sym in ["ar", "de", "el", "es", "hi", "ru", "th", "tr", "vi", "zh-CN", "iw"]:

        for _type in ["train", "dev"]:
            input_path = f'{opt.input_path}/{_type}-v2.0_{sym}_base.json'
            try:
                with open(input_path) as json_file:
                    full_doc = json.load(json_file)

                    data = full_doc['data']

                new_data = []
                total = 0
                translated = 0
                for subject in data:
                    for paragraph in subject['paragraphs']:
                        total += 1
                        if 'translated' in paragraph:
                            translated += 1

                print(f'{input_path}\t{translated/total*100}%')
            except:
                print(f'{input_path}\t0%')

