# batch (total on all devices = 16)
export BASEPATH=/home/ofri/qa_translate
export PYTHONPATH=${BASEPATH}/src
export DATAPATH=${BASEPATH}/data
export CUDA_VISIBLE_DEVICES=2,3

python ${PYTHONPATH}/train/run_qa.py \
  --model_name_or_path bert-base-multilingual-cased \
  --train_file ${DATAPATH}/parashoot/train.json \
  --validation_file ${DATAPATH}/parashoot/dev.json \
  --test_file ${DATAPATH}/parashoot/test.json \
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
  --output_dir ${BASEPATH}/exp/parashoot_reproduce \
  --run_name parashoot_reproduce \
  --overwrite_output_dir \
  --overwrite_cache \
  --num_train_epochs 5 \
  --he_metrics

python ${PYTHONPATH}/train/run_qa.py \
  --model_name_or_path bert-base-multilingual-cased \
  --train_file ${DATAPATH}/squad/train-v1.1-he_0.json \
  --validation_file ${DATAPATH}/parashoot/dev.json \
  --test_file ${DATAPATH}/parashoot/all.json \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-04 \
  --doc_stride 128 \
  --max_seq_length 512 \
  --logging_steps 5 \
  --seed 200 \
  --fp16 \
  --warmup_ratio 0.06 \
  --eval_steps 40 \
  --evaluation_strategy steps \
  --output_dir ${PYTHONPATH}/exp/train_on_he_0_test_on_parashoot_all \
  --run_name train_on_he_0_test_on_parashoot_all \
  --overwrite_output_dir \
  --overwrite_cache \
  --num_train_epochs 5 \
  --he_metrics

python ${PYTHONPATH}/train/run_qa.py \
  --model_name_or_path bert-base-multilingual-cased \
  --train_file ${DATAPATH}/squad/train-v1.1-he_1.json \
  --validation_file ${DATAPATH}/parashoot/dev.json \
  --test_file ${DATAPATH}/parashoot/all.json \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-04 \
  --doc_stride 128 \
  --max_seq_length 512 \
  --logging_steps 5 \
  --seed 200 \
  --fp16 \
  --warmup_ratio 0.06 \
  --eval_steps 40 \
  --evaluation_strategy steps \
  --output_dir ${PYTHONPATH}/exp/train_on_he_1_test_on_parashoot_all \
  --run_name train_on_he_1_test_on_parashoot_all \
  --overwrite_output_dir \
  --overwrite_cache \
  --num_train_epochs 5 \
  --he_metrics

python ${PYTHONPATH}/train/run_qa.py \
  --model_name_or_path bert-base-multilingual-cased \
  --train_file ${DATAPATH}/squad/train-v1.1-he_2.json \
  --validation_file ${DATAPATH}/parashoot/dev.json \
  --test_file ${DATAPATH}/parashoot/all.json \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-04 \
  --doc_stride 128 \
  --max_seq_length 512 \
  --logging_steps 5 \
  --seed 200 \
  --fp16 \
  --warmup_ratio 0.06 \
  --eval_steps 40 \
  --evaluation_strategy steps \
  --output_dir ${PYTHONPATH}/exp/train_on_he_2_test_on_parashoot_all \
  --run_name train_on_he_2_test_on_parashoot_all \
  --overwrite_output_dir \
  --overwrite_cache \
  --num_train_epochs 5 \
  --he_metrics

python ${PYTHONPATH}/train/run_qa.py \
  --model_name_or_path bert-base-multilingual-cased \
  --train_file ${DATAPATH}/squad/train-v1.1-he_2-sub1792.json \
  --validation_file ${DATAPATH}/parashoot/dev.json \
  --test_file ${DATAPATH}/parashoot/all.json \
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
  --output_dir ${PYTHONPATH}/exp/train_on_he_2-sub1792_test_on_parashoot_all \
  --run_name train_on_he_2-sub1792_test_on_parashoot_all \
  --overwrite_output_dir \
  --overwrite_cache \
  --num_train_epochs 5 \
  --he_metrics

python ${PYTHONPATH}/train/run_qa.py \
  --model_name_or_path bert-base-multilingual-cased \
  --train_file ${DATAPATH}/squad/train-v1.1-he_2-sub53352.json \
  --validation_file ${DATAPATH}/parashoot/dev.json \
  --test_file ${DATAPATH}/parashoot/all.json \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-04 \
  --doc_stride 128 \
  --max_seq_length 512 \
  --logging_steps 5 \
  --seed 200 \
  --fp16 \
  --warmup_ratio 0.06 \
  --eval_steps 40 \
  --evaluation_strategy steps \
  --output_dir ${PYTHONPATH}/exp/train_on_he_2-sub53352_test_on_parashoot_all \
  --run_name train_on_he_2-sub53352_test_on_parashoot_all \
  --overwrite_output_dir \
  --overwrite_cache \
  --num_train_epochs 5 \
  --he_metrics