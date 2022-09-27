import argparse
import json

from genson import SchemaBuilder

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input_json', type=str)
    opt = parser.parse_args()

    with open(opt.input_json) as json_file:
        full_doc = json.load(json_file)


        sb = SchemaBuilder()
        sb.add_object(full_doc)

        print(json.dumps(sb.to_schema(), indent=2))