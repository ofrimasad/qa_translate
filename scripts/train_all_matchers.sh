# batch (total on all devices = 16)
export PYTHONPATH=/home/ofri/qa_translate/src
export CUDA_VISIBLE_DEVICES=0,1





for lang in "ar" "de" "el" "es" "hi" # "ru" "th" "tr" "vi" "zh-CN" "iw"
do
  python ${PYTHONPATH}/train/run_qa.py \
    --model_name_or_path=bert-base-multilingual-cased \
    --train_file=/home/ofri/hebrew_qa/data/matcher_datasets_new/train-v2.0hf_${lang}_base_matcher_${lang}_enq.json \
    --validation_file=/home/ofri/hebrew_qa/data/matcher_datasets_new/dev-v2.0hf_${lang}_base_matcher_${lang}_enq.json \
    --test_file=/home/ofri/hebrew_qa/data/matcher_datasets_new/dev-v2.0hf_${lang}_base_matcher_${lang}_enq.json \
    --output_dir=/home/ofri/hebrew_qa/matcher_exp/enq_new/train_matcher_${lang} \
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
    --save_total_limit=2 \
    --load_best_model_at_end=True \
    --max_train_samples=600000 \
    --max_eval_samples=15000 \
    --max_predict_samples=15000
done