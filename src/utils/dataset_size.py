import argparse
import json
import os

from tqdm import tqdm


if __name__ == "__main__":



    with open('/home/ofri/qa_translate/data/pquad/pqa_train.json') as json_file:
        full_doc = json.load(json_file)
        data = full_doc['data']
        print(len(data))
