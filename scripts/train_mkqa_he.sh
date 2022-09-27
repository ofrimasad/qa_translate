# train alephbert-base on MKQA-he and test on SQuAD v2 dev
SQUAD_DIR=../data/squad

python ../bertserini/train/run_squad.py \
  --model_type bert \
  --model_name_or_path onlplab/alephbert-base \
  --do_train \
  --evaluate_during_training \
  --do_lower_case \
  --train_file ../data/mkqa/train-mkqa-he.json \
  --predict_file $SQUAD_DIR/dev-v2.0-he.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./tmp/alephbert_mkqa_he/ \
  --overwrite_output_dir \
  --dataset_name mkqa_en2he

