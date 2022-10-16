# batch (total on all devices = 16)
export PYTHONPATH=/home/ofri/qa_translate/src
export CUDA_VISIBLE_DEVICES=2,3
python ${PYTHONPATH}/src/train/run_qa.py \
  --model_name_or_path bert-base-multilingual-cased \
  --train_file ${PYTHONPATH}/data/parashoot/train.json \
  --validation_file ${PYTHONPATH}/data/parashoot/dev.json \
  --test_file ${PYTHONPATH}/data/parashoot/test.json \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 8 \
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
  --output_dir ${PYTHONPATH}/exp/parashoot_reproduce \
  --run_name parashoot_reproduce \
  --overwrite_output_dir \
  --overwrite_cache \
  --num_train_epochs 5