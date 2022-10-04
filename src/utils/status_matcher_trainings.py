

import argparse
import json
import os.path


import glob
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)

    opt = parser.parse_args()



    for sym in ["ar", "de", "el", "es", "hi", "ru", "th", "tr", "vi", "zh-CN", "iw"]:
        for enq in [True, False]:

            input_path = f'{opt.input_path}/{"enq_new" if enq else "non_enq_new"}/train_matcher_{sym}/predict_results.json'
            try:
                with open(input_path) as json_file:
                    full_doc = json.load(json_file)

                print(f'{input_path}\t{full_doc["test_f1"]}\t{full_doc["test_exact"]}')
            except:
                print(f'{input_path}')

