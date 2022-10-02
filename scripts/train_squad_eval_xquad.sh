# batch (total on all devices = 16)
export PYTHONPATH=/home/ofri/qa_translate
export CUDA_VISIBLE_DEVICES=0,1

for lang in "ar" "de" "el" "es" "hi" "ru" "th" "tr" "vi" "zh-CN" "iw"
do
  python ${PYTHONPATH}/src/train/run_qa.py \
    --model_name_or_path bert-base-multilingual-cased \
    --train_file=${PYTHONPATH}/data/squad/enq95/train_v2.0hf_${lang}_0.95_enq.json \
    --validation_file=${PYTHONPATH}/data/squad/enq95/dev_v2.0hf_${lang}_0.95_enq.json \
    --test_file ${PYTHONPATH}/data/xquad/xquad.${lang}.v1.1_format.json \
    --output_dir ${PYTHONPATH}/exp_xquad/train_squad_mm05_from_en_newline_test_xquad_${lang} \
    --run_name train_squad_mm05_from_en_newline_test_xquad_${lang} \
    --do_train \
    --do_eval \
    --do_predict \
    --doc_stride=128 \
    --max_seq_length=384 \
    --per_gpu_train_batch_size=4 \
    --gradient_accumulation_steps=8 \
    --learning_rate=3e-5 \
    --save_steps=-1 \
    --num_train_epochs=3.0 \
    --warmup_steps=500 \
    --evaluation_strategy steps \
    --weight_decay=0.0001 \
    --overwrite_output_dir
done


