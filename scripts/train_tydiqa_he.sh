# train alephbert-base on original SQuAD-he v2.0
TYDIQA_DIR=../data/tydiqa

python ../bertserini/train/run_squad.py \
  --model_type bert \
  --model_name_or_path onlplab/alephbert-base \
  --do_train \
  --evaluate_during_training \
  --do_lower_case \
  --train_file $TYDIQA_DIR/tydiqa-goldp-v1.1-train-he.json \
  --predict_file $TYDIQA_DIR/tydiqa-goldp-v1.1-dev-he.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 18.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./tmp/alephbert_squad2_he/ \
  --overwrite_output_dir \
  --dataset_name tydiqa_ar2he \
  --freeze_encoder

