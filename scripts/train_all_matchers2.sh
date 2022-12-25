# batch (total on all devices = 16)
export BASEPATH=/home/ofri/qa_translate
export PYTHONPATH=${BASEPATH}/src
export DATAPATH=${BASEPATH}/data
export CUDA_VISIBLE_DEVICES=2,3


for lang in "ar" "de" "el" "es" "hi" "ru" "th" "tr" "vi" "zh-CN"
do
  python ${PYTHONPATH}/train/run_qa.py \
    --model_name_or_path=bert-base-multilingual-cased \
    --train_file=${DATAPATH}/matcher_datasets_25_12_2022/train-v2.0hf_${lang}_base_matcher_${lang}_enq.json \
    --validation_file=${DATAPATH}/matcher_datasets_07_12_2022/dev-v2.0hf_${lang}_base_matcher_${lang}_enq.json \
    --test_file=${DATAPATH}/matcher_datasets_07_12_2022/dev-v2.0hf_${lang}_base_matcher_${lang}_enq.json \
    --output_dir=${BASEPATH}/matcher_exp/enq_25_12_2022/train_matcher_${lang} \
    --run_name=train_matcher_${lang} \
    --do_train \
    --do_eval \
    --do_predict \
    --doc_stride=128 \
    --max_seq_length=384 \
    --per_gpu_train_batch_size=36 \
    --gradient_accumulation_steps=8 \
    --learning_rate=3e-5 \
    --num_train_epochs=4.0 \
    --warmup_steps=500 \
    --evaluation_strategy=steps \
    --weight_decay=0.0001 \
    --overwrite_output_dir \
    --version_2_with_negative \
    --metric_for_best_model=eval_exact \
    --save_total_limit=1 \
    --load_best_model_at_end=True
done