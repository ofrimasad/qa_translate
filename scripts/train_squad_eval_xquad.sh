# batch (total on all devices = 16)
export PYTHONPATH=/home/ofri/qa_translate/src
export DATAPATH=/home/ofri/qa_translate/data
export CUDA_VISIBLE_DEVICES=2,3

for lang in "de" #"ar" "de" "el" "es" "hi"
do
  python ${PYTHONPATH}/train/run_qa.py \
    --model_name_or_path bert-base-multilingual-cased \
    --train_file=${DATAPATH}/squad/enq05n_old_match/train_v1.0hf_${lang}_0.50_enq.json \
    --validation_file=${DATAPATH}/squad/enq05/dev_v1.0hf_${lang}_0.50_enq.json \
    --test_file=${DATAPATH}/xquad/xquad.${lang}.v1.1_format.json \
    --output_dir=${PYTHONPATH}/exp_xquad_new/train_squad_05_old_match_test_xquad_${lang} \
    --run_name=train_squad_05_old_match_test_xquad_${lang} \
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


