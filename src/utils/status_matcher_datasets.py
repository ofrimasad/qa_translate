

import argparse
import json
import os.path


import glob
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)

    opt = parser.parse_args()



    for sym in ["ar", "de", "el", "es", "hi", "ru", "th", "tr", "vi", "zh-CN", "iw"]:
        for enq in ["_enq", ""]:
            for _type in ["train", "dev"]:

                input_path = f'{opt.input_path}/{_type}-v2.0hf_{sym}_base_matcher_{sym}{enq}.json'
                try:
                    with open(input_path) as json_file:
                        full_doc = json.load(json_file)

                        data = full_doc['data']

                    possible = 0
                    impossible = 0
                    for subject in data:

                        if len(subject['answers']['text']) > 0:
                            possible +=1
                        else:
                            impossible += 1


                    print(f'{input_path}\t{possible}\t{impossible}')
                except:
                    print(f'{input_path}')

