

import argparse
import json
import os.path


import glob
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)

    opt = parser.parse_args()



    for sym in ["ar", "de", "el", "es", "hi", "ru", "th", "tr", "vi", "zh-CN"]:
        for enq in [ False]:

            predict_path = f'{opt.input_path}_{sym}/predict_results.json'
            train_path = f'{opt.input_path}_{sym}/train_results.json'

            try:
                with open(predict_path) as json_file:
                    predict_doc = json.load(json_file)
                with open(train_path) as json_file:
                    train_doc = json.load(json_file)

                print(f'{sym}\t{predict_doc["test_f1"]:.2f}\t{predict_doc["test_exact_match"]:.2f}\t{train_doc["train_samples"]}')
            except:
                print(f'{sym}')

