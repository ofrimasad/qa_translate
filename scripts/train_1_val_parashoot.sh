# batch (total on all devices = 16)
export PYTHONPATH=/home/ofri/qa_translate/src
CUDA_VISIBLE_DEVICES=0,1,2,3 python /home/ofri/qa_translate/src/train/run_qa.py \
  --model_name_or_path bert-base-multilingual-cased \
  --train_file /home/ofri/qa_translate/data/squad/train-v1.1-he_1.json \
  --validation_file /home/ofri/qa_translate/data/parashoot/dev.json \
  --test_file /home/ofri/qa_translate/data/parashoot/test.json \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-04 \
  --doc_stride 128 \
  --max_seq_length 512 \
  --logging_steps 5 \
  --seed 200 \
  --fp16 \
  --warmup_ratio 0.06 \
  --eval_steps 40 \
  --evaluation_strategy steps \
  --output_dir ./parashoot_mbert_lr_5e-05_bsz_32_max_epochs_5_seed_200 \
  --run_name parashoot_mbert_lr_5e-05_bsz_32_max_epochs_5_seed_200 \
  --overwrite_output_dir \
  --overwrite_cache \
  --num_train_epochs 5