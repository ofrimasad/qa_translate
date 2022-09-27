# train roberta-base on original SQuAD v2.0
BASE_DIR=/home/ofri/qa_translate

python $BASE_DIR/src/train/train.py \
  --run_name baseline_robareta_squad2_en \
  --lang_model roberta-base \
  --train_filename train-v2.0.json \
  --dev_filename dev-v2.0.json


