
CUDA_VISIBLE_DEVICES=0 python translate_squad.py -i ../data/squad/train-v2.0.json -o ../data/squad/train-v2.0-he_0.json --no_markers
CUDA_VISIBLE_DEVICES=0 python translate_squad.py -i ../data/squad/dev-v2.0.json -o ../data/squad/dev-v2.0-he_0.json --no_markers
CUDA_VISIBLE_DEVICES=0 python translate_squad.py -i ../data/squad/train-v2.0.json -o ../data/squad/train-v2.0-he_1.json
CUDA_VISIBLE_DEVICES=0 python translate_squad.py -i ../data/squad/dev-v2.0.json -o ../data/squad/dev-v2.0-he_1.json
CUDA_VISIBLE_DEVICES=0 python translate_squad.py -i ../data/squad/train-v2.0.json -o ../data/squad/train-v2.0-he_2.json --replace
CUDA_VISIBLE_DEVICES=0 python translate_squad.py -i ../data/squad/dev-v2.0.json -o ../data/squad/dev-v2.0-he_2.json --replace