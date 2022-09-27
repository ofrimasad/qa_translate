# train the original bert-base-uncased on original SQuAD v2.0
SQUAD_DIR=../data/squad

python ../bertserini/train/run_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --evaluate_during_training \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v2.0.json \
  --predict_file $SQUAD_DIR/dev-v2.0.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./tmp/bert_base_squad2/ \
  --overwrite_output_dir \
  --dataset_name squad_en

