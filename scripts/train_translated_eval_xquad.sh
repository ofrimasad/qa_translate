# batch (total on all devices = 16)
export BASEPATH=/home/ofri/qa_translate
export PYTHONPATH=${BASEPATH}/src
export DATAPATH=${BASEPATH}/data
export CUDA_VISIBLE_DEVICES=2,3

for lang in "de" "el" "ru" "tr" "ar" "vi" "th" "hi" "zh-CN"
do
  python ${PYTHONPATH}/train/run_qa.py \
    --model_name_or_path bert-base-multilingual-cased \
    --train_file ${DATAPATH}/squad/v1.1_translated/train-v1.1_${lang.json \
    --validation_file ${DATAPATH}/xquad/xquad.$lang.v1.1_format.json \
    --test_file ${DATAPATH}/xquad/xquad.$lang.v1.1_format.json \
    --output_dir ${BASEPATH}/exp_xquad/train_squad_test_xquad_$lang \
    --run_name train_squad_test_xquad_$lang_01 \
    --do_train \
    --do_eval \
    --do_predict \
    --doc_stride 128 \
    --max_seq_length 384 \
    --per_gpu_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-5 \
    --save_steps -1 \
    --num_train_epochs 3.0 \
    --warmup_steps 500 \
    --evaluation_strategy steps \
    --weight_decay 0.0001 \
    --overwrite_output_dir

  echo "#########################################"
  echo "#########################################"
  echo "#########################################"
done

